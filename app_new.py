import os
import time
import json
import io
import re
import boto3
import pandas as pd
import streamlit as st

# Base directory for absolute path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set up Streamlit page configuration
st.set_page_config(page_title="Sigma AI-Powered Data Pipeline Dashboard", layout="wide")

# CSS Styling Injection (accent color #0096FF, minimum 16px font size)
st.markdown("""
<style>
    /* Force 16px minimum font size for Smart TV readability */
    html, body, p, li, label, .stSelectbox, .stTextInput, .stButton, [data-testid="stMetricValue"] {
        font-size: 16px !important;
    }
    
    /* Title and Header Accent Colors */
    h1, h2, h3, [data-testid="stHeader"] {
        color: #0096FF !important;
        font-weight: bold;
    }
    
    /* Metric Label Colors */
    [data-testid="stMetricLabel"] {
        color: #0096FF !important;
        font-size: 16px !important;
        font-weight: 500;
    }
    
    /* Button Custom styling */
    div.stButton > button:first-child {
        background-color: #0096FF !important;
        color: white !important;
        border: 1px solid #0096FF !important;
        border-radius: 4px;
        font-size: 16px !important;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #0080E5 !important;
        border-color: #0080E5 !important;
        color: white !important;
    }
    
    /* Tab highlight color and border styling */
    button[data-baseweb="tab"] {
        font-size: 18px !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #0096FF !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0096FF !important;
        border-bottom-color: #0096FF !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# SIDEBAR — Generic AWS Config & Unique Suffixes
# ----------------------------------------------------
st.sidebar.title("⚙️ Pipeline Settings")

# AWS Credentials Info Message as requested
st.sidebar.info(
    "💡 **AWS Credentials Note:**\n"
    "If you are using a terminal-based undeployed app and you have already configured AWS credentials in your terminal, "
    "please **ignore / leave these fields blank**. Otherwise, if this is a deployed application, please enter your credentials below."
)

# Sidebar credential inputs
aws_access_key = st.sidebar.text_input("AWS Access Key ID", type="password", help="Leave blank to use environment/local profile credentials")
aws_secret_key = st.sidebar.text_input("AWS Secret Access Key", type="password", help="Leave blank to use environment/local profile credentials")
aws_region = st.sidebar.text_input("AWS Region", value="us-east-1", help="Default region is us-east-1")

# Sidebar unique project suffix mapping
unique_name_input = st.sidebar.text_input(
    "Unique Team/Project Name", 
    value="nexus", 
    help="Replaces 'nexus' across all S3 buckets, Glue jobs, databases, and Athena tables to avoid global name conflicts."
).strip().lower()

# Clean unique name to ensure valid resource naming structure
unique_name = re.sub(r'[^a-z0-9\-]', '', unique_name_input)
if not unique_name:
    unique_name = "nexus"

# Dynamic Resource Definitions
BUCKET_NAME = f"sigma-{unique_name}-bucket"
GLUE_JOB = f"sigma-{unique_name}-etl"
ATHENA_DB = f"sigma_{unique_name}_db".replace("-", "_")
ATHENA_TABLE_ORDERS = f"sigma_{unique_name}_orders".replace("-", "_")
ATHENA_TABLE_CUSTOMERS = f"sigma_{unique_name}_customers".replace("-", "_")
ATHENA_TABLE_PRODUCTS = f"sigma_{unique_name}_products".replace("-", "_")
GLUE_ROLE = "SigmaGlueServiceRole"

# Display constructed active resources in sidebar for verification
st.sidebar.write("---")
st.sidebar.markdown(f"""
### 📦 Target AWS Resources
- **S3 Bucket:** `{BUCKET_NAME}`
- **Glue Job:** `{GLUE_JOB}`
- **Athena Database:** `{ATHENA_DB}`
- **Athena Tables:** 
  - `{ATHENA_TABLE_ORDERS}`
  - `{ATHENA_TABLE_CUSTOMERS}`
  - `{ATHENA_TABLE_PRODUCTS}`
- **Glue IAM Role:** `{GLUE_ROLE}`
""")

# ----------------------------------------------------
# AWS Session & Client Factories
# ----------------------------------------------------
def get_boto3_session():
    if aws_access_key and aws_secret_key:
        return boto3.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region or 'us-east-1'
        )
    else:
        return boto3.Session(region_name=aws_region or 'us-east-1')

def get_aws_client(service_name):
    session = get_boto3_session()
    return session.client(service_name)

# Helper function for Bedrock converse() API call
def ask_bedrock(prompt: str) -> str:
    try:
        bedrock = get_aws_client('bedrock-runtime')
        response = bedrock.converse(
            modelId="us.amazon.nova-lite-v1:0",
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 500, "temperature": 0.0}
        )
        return response["output"]["message"]["content"][0]["text"].strip()
    except Exception as e:
        return f"Error calling Bedrock: {str(e)}"

def run_athena_query(query: str, database: str = "default") -> pd.DataFrame:
    athena = get_aws_client('athena')
    response = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={'Database': database},
        ResultConfiguration={'OutputLocation': f"s3://{BUCKET_NAME}/athena-results/"}
    )
    query_execution_id = response['QueryExecutionId']
    
    while True:
        status_resp = athena.get_query_execution(QueryExecutionId=query_execution_id)
        state = status_resp['QueryExecution']['Status']['State']
        if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            if state == 'FAILED':
                reason = status_resp['QueryExecution']['Status'].get('StateChangeReason', 'Unknown Athena error')
                raise Exception(f"Athena query failed: {reason}")
            if state == 'CANCELLED':
                raise Exception("Athena query was cancelled.")
            break
        time.sleep(2)
        
    # Check if the query is a DDL/DML that doesn't return structured rows (e.g. MSCK, CREATE, DROP, ALTER)
    cleaned_query = query.strip().upper()
    if not (cleaned_query.startswith("SELECT") or cleaned_query.startswith("SHOW") or cleaned_query.startswith("DESCRIBE")):
        return pd.DataFrame([{"status": "Query executed successfully."}])
        
    results = athena.get_query_results(QueryExecutionId=query_execution_id)
    
    cols = [c["Label"] for c in results["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]]
    all_rows = results["ResultSet"]["Rows"]
    # Only skip Rows[0] if it actually matches the column names (SELECT header row)
    if all_rows and [f.get("VarCharValue", "") for f in all_rows[0]["Data"]] == cols:
        all_rows = all_rows[1:]
    data = [[field.get("VarCharValue", "") for field in row["Data"]] for row in all_rows]
    return pd.DataFrame(data, columns=cols)

# Format monetary values to Indian Rupees with commas and no decimals
def format_monetary_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    monetary_keywords = ['amount', 'revenue', 'total_sales', 'sales', 'price', 'avg_amount', 'sum_amount', 'total_revenue']
    monetary_cols = [col for col in df_copy.columns if any(kw in col.lower() for kw in monetary_keywords)]
    
    for col in monetary_cols:
        try:
            df_copy[col] = df_copy[col].apply(
                lambda x: f"₹{int(round(float(x))):,}" if pd.notnull(x) and str(x).strip() != '' else x
            )
        except Exception:
            pass
    return df_copy

# Streamlit App Layout
st.title("Sigma Pipeline Dashboard — Modern Operations Control")
st.write("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔧 Setup Pipeline", 
    "📦 Daily Load", 
    "🔍 Ask Your Data", 
    "📊 Pipeline Health"
])

# ----------------------------------------------------
# Session State Initializations for Persistent UI Elements
# ----------------------------------------------------
if "deploy_steps" not in st.session_state:
    st.session_state["deploy_steps"] = []

# ----------------------------------------------------
# TAB 1 — Setup Pipeline
# ----------------------------------------------------
with tab1:
    st.header(f"Sigma {unique_name.upper()} — Pipeline Setup")
    
    # 8-step Deployment Plan Descriptions
    steps_definitions = [
        {
            "title": "1. Create S3 Bucket", 
            "desc": f"Validates and provisions globally unique bucket `s3://{BUCKET_NAME}` in region `{aws_region or 'us-east-1'}` (skips CreateBucketConfiguration default)."
        },
        {
            "title": "2. Upload Glue ETL Script", 
            "desc": f"Reads the local compiled script `glue_scripts/etl.py` and uploads it to S3 at `s3://{BUCKET_NAME}/glue-scripts/etl.py`."
        },
        {
            "title": "3. Upload Dimension Tables", 
            "desc": f"Uploads references `customers.csv` and `products.csv` to both raw and processed locations to satisfy Glue reference copies."
        },
        {
            "title": "4. Configure Glue ETL Job", 
            "desc": f"Deletes existing job and provisions `{GLUE_JOB}` on version '1.0' (Python Shell, pandas dependency, capacity 0.0625, timeout 10)."
        },
        {
            "title": "5. Setup Athena database schema", 
            "desc": f"Issues DDL against standard catalog to configure schema `{ATHENA_DB}`."
        },
        {
            "title": "6. Setup Orders Table Partition DDL", 
            "desc": f"Configures partition schema for table `{ATHENA_TABLE_ORDERS}` pointing to S3 location `processed/orders/`."
        },
        {
            "title": "7. Provision Customers Table", 
            "desc": f"Drops existing table and registers dimension table `{ATHENA_TABLE_CUSTOMERS}` under `processed/customers/`."
        },
        {
            "title": "8. Provision Products Table", 
            "desc": f"Drops existing table and registers dimension table `{ATHENA_TABLE_PRODUCTS}` under `processed/products/`."
        }
    ]

    st.markdown("### 📋 Pipeline Deployment Strategy & Setup Details")
    st.write("The deployment will perform the following pipeline configurations automatically:")
    
    # Render detailed descriptions of what the deployment will do
    for step in steps_definitions:
        st.markdown(f"**🔹 {step['title']}**\n*{step['desc']}*")
    st.write("---")

    if st.button("🚀 Deploy Pipeline"):
        s3_client = get_aws_client('s3')
        glue_client = get_aws_client('glue')
        sts_client = get_aws_client('sts')
        
        # Reset and initialize deployment steps in session state
        st.session_state["deploy_steps"] = [
            {"title": step["title"], "desc": step["desc"], "status": "running", "message": "Deploying step..."}
            for step in steps_definitions
        ]
        
        # Helper to update persistent state live
        def update_step_state(idx, status, message):
            st.session_state["deploy_steps"][idx]["status"] = status
            st.session_state["deploy_steps"][idx]["message"] = message
            st.rerun()

        # Step 1: Create S3 bucket if not exists
        try:
            try:
                s3_client.head_bucket(Bucket=BUCKET_NAME)
                st.session_state["deploy_steps"][0]["status"] = "success"
                st.session_state["deploy_steps"][0]["message"] = f"Bucket `s3://{BUCKET_NAME}` already exists and is active. ✅"
            except s3_client.exceptions.ClientError:
                s3_client.create_bucket(Bucket=BUCKET_NAME)
                st.session_state["deploy_steps"][0]["status"] = "success"
                st.session_state["deploy_steps"][0]["message"] = f"Bucket `s3://{BUCKET_NAME}` created successfully. ✅"
        except Exception as e:
            st.session_state["deploy_steps"][0]["status"] = "error"
            st.session_state["deploy_steps"][0]["message"] = f"Failed: {str(e)} ❌"
            st.rerun()
            st.stop()
            
        # Step 2: Upload glue_scripts/etl.py to S3
        try:
            script_path = os.path.join(BASE_DIR, "glue_scripts", "etl.py")
            with open(script_path, "r", encoding="utf-8") as f:
                etl_code = f.read()
            s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key="glue-scripts/etl.py",
                Body=etl_code.encode("utf-8")
            )
            st.session_state["deploy_steps"][1]["status"] = "success"
            st.session_state["deploy_steps"][1]["message"] = f"Uploaded script to `s3://{BUCKET_NAME}/glue-scripts/etl.py`. ✅"
        except Exception as e:
            st.session_state["deploy_steps"][1]["status"] = "error"
            st.session_state["deploy_steps"][1]["message"] = f"Failed: {str(e)} ❌"
            st.rerun()
            st.stop()
            
        # Step 3: Upload customers.csv and products.csv to raw/ and processed/ unchanged
        try:
            cust_path = os.path.join(BASE_DIR, "data", "customers.csv")
            prod_path = os.path.join(BASE_DIR, "data", "products.csv")
            cust_df = pd.read_csv(cust_path)
            prod_df = pd.read_csv(prod_path)
            
            cust_bytes = cust_df.to_csv(index=False).encode("utf-8")
            prod_bytes = prod_df.to_csv(index=False).encode("utf-8")
            
            # Write to raw S3 prefix
            s3_client.put_object(Bucket=BUCKET_NAME, Key="raw/customers.csv", Body=cust_bytes)
            s3_client.put_object(Bucket=BUCKET_NAME, Key="raw/products.csv", Body=prod_bytes)
            
            # Write to processed S3 prefix (for Athena queries)
            s3_client.put_object(Bucket=BUCKET_NAME, Key="processed/customers/customers.csv", Body=cust_bytes)
            s3_client.put_object(Bucket=BUCKET_NAME, Key="processed/products/products.csv", Body=prod_bytes)
            
            st.session_state["deploy_steps"][2]["status"] = "success"
            st.session_state["deploy_steps"][2]["message"] = "Dimension tables successfully uploaded to both raw/ and processed/ namespaces. ✅"
        except Exception as e:
            st.session_state["deploy_steps"][2]["status"] = "error"
            st.session_state["deploy_steps"][2]["message"] = f"Failed: {str(e)} ❌"
            st.rerun()
            st.stop()
            
        # Step 4: Create Glue job (Always delete first to avoid stale configurations)
        try:
            try:
                glue_client.delete_job(JobName=GLUE_JOB)
            except glue_client.exceptions.EntityNotFoundException:
                pass
            
            # Find Account ID for exact IAM role ARN construction
            try:
                account_id = sts_client.get_caller_identity()["Account"]
                role_arn = f"arn:aws:iam::{account_id}:role/{GLUE_ROLE}"
            except Exception:
                role_arn = GLUE_ROLE
                
            glue_client.create_job(
                Name=GLUE_JOB,
                Role=role_arn,
                Command={
                    "Name": "pythonshell",
                    "ScriptLocation": f"s3://{BUCKET_NAME}/glue-scripts/etl.py",
                    "PythonVersion": "3.9"
                },
                DefaultArguments={
                    "--additional-python-modules": "pandas"
                },
                MaxRetries=0,
                Timeout=10,
                MaxCapacity=0.0625,
                GlueVersion="1.0",
                ExecutionProperty={
                    "MaxConcurrentRuns": 5
                }
            )
            st.session_state["deploy_steps"][3]["status"] = "success"
            st.session_state["deploy_steps"][3]["message"] = f"Re-created Glue Python Shell job '{GLUE_JOB}' on v1.0. ✅"
        except Exception as e:
            st.session_state["deploy_steps"][3]["status"] = "error"
            st.session_state["deploy_steps"][3]["message"] = f"Failed: {str(e)} ❌"
            st.rerun()
            st.stop()
            
        # Step 5: Create Athena Database
        try:
            run_athena_query(f"CREATE SCHEMA IF NOT EXISTS {ATHENA_DB}", database="default")
            st.session_state["deploy_steps"][4]["status"] = "success"
            st.session_state["deploy_steps"][4]["message"] = f"Database `{ATHENA_DB}` metadata registered successfully. ✅"
        except Exception as e:
            st.session_state["deploy_steps"][4]["status"] = "error"
            st.session_state["deploy_steps"][4]["message"] = f"Failed: {str(e)} ❌"
            st.rerun()
            st.stop()
            
        # Step 6: Create orders partition external table in Athena
        try:
            orders_ddl = f"""
            CREATE EXTERNAL TABLE IF NOT EXISTS {ATHENA_DB}.{ATHENA_TABLE_ORDERS} (
                order_id STRING,
                customer_id STRING,
                product_id STRING,
                quantity INT,
                amount DOUBLE,
                status STRING,
                payment_method STRING,
                city STRING,
                created_at STRING,
                processed_at STRING,
                is_high_value STRING
            )
            PARTITIONED BY (date STRING)
            ROW FORMAT DELIMITED
            FIELDS TERMINATED BY ','
            STORED AS TEXTFILE
            LOCATION 's3://{BUCKET_NAME}/processed/orders/'
            TBLPROPERTIES ('skip.header.line.count'='1')
            """
            run_athena_query(orders_ddl, database=ATHENA_DB)
            st.session_state["deploy_steps"][5]["status"] = "success"
            st.session_state["deploy_steps"][5]["message"] = f"Table metadata `{ATHENA_TABLE_ORDERS}` mapped to bucket namespace. ✅"
        except Exception as e:
            st.session_state["deploy_steps"][5]["status"] = "error"
            st.session_state["deploy_steps"][5]["message"] = f"Failed: {str(e)} ❌"
            st.rerun()
            st.stop()
            
        # Step 7: Create customers table (Drop if exists then create)
        try:
            run_athena_query(f"DROP TABLE IF EXISTS {ATHENA_DB}.{ATHENA_TABLE_CUSTOMERS}", database=ATHENA_DB)
            
            cust_ddl = f"""
            CREATE EXTERNAL TABLE {ATHENA_DB}.{ATHENA_TABLE_CUSTOMERS} (
                customer_id STRING,
                name STRING,
                email STRING,
                phone STRING,
                city STRING,
                tier STRING,
                signup_date STRING
            )
            ROW FORMAT DELIMITED
            FIELDS TERMINATED BY ','
            STORED AS TEXTFILE
            LOCATION 's3://{BUCKET_NAME}/processed/customers/'
            TBLPROPERTIES ('skip.header.line.count'='1')
            """
            run_athena_query(cust_ddl, database=ATHENA_DB)
            st.session_state["deploy_steps"][6]["status"] = "success"
            st.session_state["deploy_steps"][6]["message"] = f"Dimension table `{ATHENA_TABLE_CUSTOMERS}` provisioned and schema mapped. ✅"
        except Exception as e:
            st.session_state["deploy_steps"][6]["status"] = "error"
            st.session_state["deploy_steps"][6]["message"] = f"Failed: {str(e)} ❌"
            st.rerun()
            st.stop()
            
        # Step 8: Create products table (Drop if exists then create)
        try:
            run_athena_query(f"DROP TABLE IF EXISTS {ATHENA_DB}.{ATHENA_TABLE_PRODUCTS}", database=ATHENA_DB)
            
            prod_ddl = f"""
            CREATE EXTERNAL TABLE {ATHENA_DB}.{ATHENA_TABLE_PRODUCTS} (
                product_id STRING,
                name STRING,
                category STRING,
                price DOUBLE,
                stock_quantity INT,
                is_active STRING
            )
            ROW FORMAT DELIMITED
            FIELDS TERMINATED BY ','
            STORED AS TEXTFILE
            LOCATION 's3://{BUCKET_NAME}/processed/products/'
            TBLPROPERTIES ('skip.header.line.count'='1')
            """
            run_athena_query(prod_ddl, database=ATHENA_DB)
            st.session_state["deploy_steps"][7]["status"] = "success"
            st.session_state["deploy_steps"][7]["message"] = f"Dimension table `{ATHENA_TABLE_PRODUCTS}` provisioned and schemas complete. ✅"
            st.balloons()
            st.rerun()
        except Exception as e:
            st.session_state["deploy_steps"][7]["status"] = "error"
            st.session_state["deploy_steps"][7]["message"] = f"Failed: {str(e)} ❌"
            st.rerun()
            st.stop()

    # Renders the persistent step status from session state (Retained across tab changes!)
    if st.session_state["deploy_steps"]:
        st.markdown("### 🚀 Deployment Execution Output")
        for step in st.session_state["deploy_steps"]:
            if step["status"] == "success":
                st.success(f"**{step['title']}**\n{step['message']}")
            elif step["status"] == "error":
                st.error(f"**{step['title']}**\n{step['message']}")
            elif step["status"] == "running":
                st.info(f"**{step['title']}**\n⏳ {step['message']}")

# ----------------------------------------------------
# TAB 2 — Daily Load
# ----------------------------------------------------
with tab2:
    st.header("Daily Pipeline Load Operations")
    
    DAYS_MAP = {
        "Day 1 — 2026-05-01": ("2026-05-01", "orders_day1.csv"),
        "Day 2 — 2026-05-02": ("2026-05-02", "orders_day2.csv"),
        "Day 3 — 2026-05-03": ("2026-05-03", "orders_day3.csv"),
        "Day 4 — 2026-05-04": ("2026-05-04", "orders_day4.csv"),
        "Day 5 — 2026-05-05": ("2026-05-05", "orders_day5.csv"),
    }
    
    selected_day = st.selectbox("Select Day to Load", list(DAYS_MAP.keys()))
    partition_val, local_filename = DAYS_MAP[selected_day]
    
    if st.button(f"▶️ Run ETL for {selected_day}"):
        s3_client = get_aws_client('s3')
        glue_client = get_aws_client('glue')
        
        # Step 1: Upload the local raw csv file to S3
        with st.spinner(f"Uploading local file {local_filename} to S3 raw orders path..."):
            try:
                raw_file_path = os.path.join(BASE_DIR, "data", local_filename)
                df_raw = pd.read_csv(raw_file_path)
                raw_bytes = df_raw.to_csv(index=False).encode("utf-8")
                s3_client.put_object(
                    Bucket=BUCKET_NAME,
                    Key=f"raw/orders/date={partition_val}/orders.csv",
                    Body=raw_bytes
                )
                st.success(f"Successfully uploaded raw orders to s3://{BUCKET_NAME}/raw/orders/date={partition_val}/orders.csv")
            except Exception as e:
                st.error(f"Failed to upload raw CSV to S3: {str(e)}")
                st.stop()
                
        # Step 2: Trigger AWS Glue Python Shell job run
        with st.spinner("Starting AWS Glue Python Shell ETL Job..."):
            try:
                job_run = glue_client.start_job_run(
                    JobName=GLUE_JOB,
                    Arguments={
                        "--job_type": "orders",
                        "--bucket_name": BUCKET_NAME,
                        "--date_partition": partition_val
                    }
                )
                run_id = job_run["JobRunId"]
                st.info(f"Glue job started successfully. Run ID: {run_id}")
            except Exception as e:
                st.error(f"Failed to trigger AWS Glue job: {str(e)}")
                st.stop()
                
        # Step 3 & 4: Poll Glue Job execution
        succeeded = False
        err_msg = ""
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        max_polls = 40
        
        for poll in range(1, max_polls + 1):
            resp = glue_client.get_job_run(JobName=GLUE_JOB, RunId=run_id)
            state = resp["JobRun"]["JobRunState"]
            status_text.text(f"Polling Glue Run Status... [{state}] (Attempt {poll}/{max_polls})")
            progress_bar.progress(poll / max_polls)
            
            if state in ["SUCCEEDED", "FAILED", "STOPPED", "TIMEOUT"]:
                if state == "SUCCEEDED":
                    succeeded = True
                else:
                    err_msg = resp["JobRun"].get("ErrorMessage", f"Job ended with state: {state}")
                break
            time.sleep(3)
            
        progress_bar.empty()
        status_text.empty()
        
        if not succeeded:
            st.error(f"❌ Glue Job Run failed: {err_msg if err_msg else 'Timeout exceeded'}")
            st.stop()
            
        st.success("✅ Glue job run finished successfully.")
        
        # Synchronize newly written partitions to Athena metadata store
        with st.spinner("Synchronizing newly ingested S3 partitions using MSCK REPAIR TABLE..."):
            try:
                run_athena_query(f"MSCK REPAIR TABLE {ATHENA_DB}.{ATHENA_TABLE_ORDERS}", database=ATHENA_DB)
                st.success("Athena partitions successfully synchronized. ✅")
            except Exception as e:
                st.error(f"Failed to update Athena partitions: {str(e)}")
                st.stop()
                
        # Step 6 & 7: Read quality report and display metrics
        with st.spinner("Retrieving pipeline quality report..."):
            try:
                report_obj = s3_client.get_object(
                    Bucket=BUCKET_NAME,
                    Key=f"reports/quality_report_{partition_val}.json"
                )
                report = json.loads(report_obj["Body"].read().decode("utf-8"))
            except Exception as e:
                st.error(f"Failed to load quality report from S3: {str(e)}")
                st.stop()
                
        st.markdown("### 📊 Ingestion Quality Metrics")
        m_cols = st.columns(6)
        m_cols[0].metric("Input Rows", f"{report['input_rows']:,}")
        m_cols[1].metric("Output Rows", f"{report['output_rows']:,}")
        m_cols[2].metric("Rows Dropped", f"{report['rows_dropped']:,}")
        m_cols[3].metric("Null Customer IDs", f"{report['null_customer_ids']:,}")
        m_cols[4].metric("Negative Amounts", f"{report['negative_amounts']:,}")
        m_cols[5].metric("Duplicate Order IDs", f"{report['duplicate_order_ids']:,}")
        
        # Step 8: Trigger warnings if any issue count is non-zero
        issues_detected = report['null_customer_ids'] > 0 or report['negative_amounts'] > 0 or report['duplicate_order_ids'] > 0
        if issues_detected:
            st.warning("⚠️ Data quality issues were identified and auto-remediated during ingestion!")
            
        # Step 9: Ask Bedrock for executive recommendations
        with st.spinner("Generating AI pipeline quality report analysis..."):
            analysis_prompt = f"""
            Analyze this pipeline health and quality ingestion report:
            Ingestion Date: {report['date']}
            Original Input Rows: {report['input_rows']}
            Cleaned Output Rows: {report['output_rows']}
            Null Customer IDs dropped: {report['null_customer_ids']}
            Negative monetary amounts fixed: {report['negative_amounts']}
            Duplicate Order IDs removed: {report['duplicate_order_ids']}
            Total Rows Dropped: {report['rows_dropped']}
            
            Return exactly a health categorization header in capitals (HEALTHY, WARNING, or CRITICAL) followed by one concise technical action recommendation. Keep the entire response under 80 words total. Do not use markdown syntax.
            """
            ai_recommendation = ask_bedrock(analysis_prompt)
            st.markdown("### 🤖 Bedrock Copilot Recommendation")
            st.info(ai_recommendation)

# ----------------------------------------------------
# TAB 3 — Ask Your Data
# ----------------------------------------------------
with tab3:
    st.header("🔍 Ask Your Data — Natural Language SQL Assistant")
    st.write("Enter questions about your logistics operations, and the Bedrock AI model will translate it into optimized Athena queries.")
    
    # Session State Staging logic for persistent input keys (Bug #9 & #10)
    if "_qq_value" in st.session_state:
        st.session_state["nl_question_input"] = st.session_state.pop("_qq_value")
        
    QUICK_QUESTIONS = [
        "Top 5 cities by revenue",
        "Daily order trend",
        "High value orders per day",
        "Top 3 payment methods by order count",
        "Average order amount by city"
    ]
    
    # Render quick-question buttons in columns
    qq_cols = st.columns(5)
    for idx, question in enumerate(QUICK_QUESTIONS):
        if qq_cols[idx].button(question, key=f"qq_{idx}"):
            st.session_state["_qq_value"] = question
            st.rerun()
            
    # Text input with key persistent setting (Bug #10)
    user_question = st.text_input(
        "Ask a question in plain English:", 
        key="nl_question_input", 
        placeholder="e.g. Find the top 3 high value customers in Mumbai"
    )
    
    if st.button("🔍 Generate SQL"):
        if user_question:
            with st.spinner("Synthesizing Athena SQL query using Amazon Bedrock..."):
                sql_generation_prompt = f"""
                You are a senior data engineer and a Presto/Athena SQL expert.
                Translate the user's natural language request into a single syntactically correct Athena query.
                
                The database name is: {ATHENA_DB}
                
                Database Schema:
                1. Table: {ATHENA_DB}.{ATHENA_TABLE_ORDERS}
                   Columns: order_id STRING, customer_id STRING, product_id STRING, quantity INT, amount DOUBLE, status STRING, payment_method STRING, city STRING, created_at STRING, processed_at STRING, is_high_value STRING, date STRING
                2. Table: {ATHENA_DB}.{ATHENA_TABLE_CUSTOMERS}
                   Columns: customer_id STRING, name STRING, email STRING, phone STRING, city STRING, tier STRING, signup_date STRING
                3. Table: {ATHENA_DB}.{ATHENA_TABLE_PRODUCTS}
                   Columns: product_id STRING, name STRING, category STRING, price DOUBLE, stock_quantity INT, is_active STRING
                   
                Rules:
                - Output only the raw SQL. NEVER include markdown fences (like ```sql), backticks, comments, or any conversational text.
                - For simple SELECT queries, always limit the result set to 100 rows (LIMIT 100).
                - For aggregation queries involving GROUP BY, SUM, or COUNT, do NOT append any LIMIT.
                - For DDL, SHOW, or DESCRIBE statements, NEVER append a LIMIT clause as it will fail execution.
                - Always wrap SUM(amount) or AVG(amount) aggregations in CAST(ROUND(...) AS BIGINT) to avoid standard scientific notations (e.g. 2.94E8).
                - Use fully qualified table names (e.g. {ATHENA_DB}.{ATHENA_TABLE_ORDERS}).
                
                User Request: "{user_question}"
                """
                
                generated_sql = ask_bedrock(sql_generation_prompt)
                
                # Cleanup markdown fences if present
                generated_sql = generated_sql.replace("```sql", "").replace("```", "").strip()
                
                # Post-process generated SQL — strip rogue LIMIT from non-SELECT statements (Bug #8)
                first_word = generated_sql.split()[0].upper() if generated_sql.split() else ""
                if first_word != "SELECT":
                    generated_sql = re.sub(r'\bLIMIT\s+\d+\b', '', generated_sql, flags=re.IGNORECASE).strip()
                    
                st.session_state["generated_sql"] = generated_sql
                
                # Reset old queries
                if "query_results" in st.session_state:
                    del st.session_state["query_results"]
        else:
            st.error("Please enter a question or select a quick question first.")
            
    # Display and run generated query if it exists
    if "generated_sql" in st.session_state:
        st.markdown("### Generated Athena SQL Query")
        st.code(st.session_state["generated_sql"], language="sql")
        
        if st.button("▶️ Run on Athena"):
            with st.spinner("Executing query on Amazon Athena..."):
                try:
                    df_results = run_athena_query(st.session_state["generated_sql"], database=ATHENA_DB)
                    st.session_state["query_results"] = df_results
                except Exception as e:
                    st.error(f"Athena query execution failed: {str(e)}")
                    
    # Display query results with Rupee formatting and Bedrock summary
    if "query_results" in st.session_state:
        st.markdown("### Query Results")
        
        df_display = format_monetary_columns(st.session_state["query_results"])
        st.dataframe(df_display, use_container_width=True)
        
        # Ask Bedrock to summarize the query results
        with st.spinner("AI Analysis of result set..."):
            summary_prompt = f"""
            Analyze the following dataset results:
            {st.session_state["query_results"].head(10).to_string(index=False)}
            
            Provide a clear, human-readable summary of these findings in exactly one plain-English sentence. Do not include markdown.
            """
            results_summary = ask_bedrock(summary_prompt)
            st.markdown("### 🤖 Bedrock Insight Summary")
            st.info(results_summary)

# ----------------------------------------------------
# TAB 4 — Pipeline Health
# ----------------------------------------------------
with tab4:
    st.header("📊 Pipeline & Business Health Monitoring")
    st.write("Aggregated daily telemetry retrieved directly from Athena tables.")
    
    if st.button("🔄 Load Health Dashboard"):
        with st.spinner("Querying Daily Business Ingestion Telemetry from Athena..."):
            health_sql = f"""
            SELECT date, 
                   COUNT(*) AS orders, 
                   CAST(ROUND(SUM(amount)) AS BIGINT) AS revenue 
            FROM {ATHENA_DB}.{ATHENA_TABLE_ORDERS} 
            GROUP BY date 
            ORDER BY date
            """
            
            try:
                df_health = run_athena_query(health_sql, database=ATHENA_DB)
                
                if not df_health.empty:
                    # Convert to proper types for plotting
                    df_health['revenue'] = pd.to_numeric(df_health['revenue'])
                    df_health['orders'] = pd.to_numeric(df_health['orders'])
                    
                    st.markdown("### Telemetry Metrics Overview")
                    t_cols = st.columns(3)
                    
                    total_orders = df_health['orders'].sum()
                    total_revenue = df_health['revenue'].sum()
                    days_loaded = len(df_health)
                    
                    t_cols[0].metric("Total Orders Processed", f"{int(total_orders):,}")
                    t_cols[1].metric("Total Revenue Transacted", f"₹{int(total_revenue):,}")
                    t_cols[2].metric("Days Loaded", f"{days_loaded}")
                    
                    st.markdown("### Daily Revenue Trend (₹)")
                    st.bar_chart(df_health, x='date', y='revenue', color='#0096FF')
                    
                    st.markdown("### Daily Order Volume Trend")
                    st.bar_chart(df_health, x='date', y='orders', color='#0096FF')
                    
                    # Generate Executive Summary using Bedrock
                    with st.spinner("Drafting AI Executive Summary..."):
                        exec_summary_prompt = f"""
                        Analyze this daily pipeline ingestion summary dataset:
                        {df_health.to_string(index=False)}
                        
                        Write a professional executive overview regarding the business metrics and the technical pipeline stability.
                        The summary must be exactly three sentences. Do not use markdown tags or headings.
                        """
                        ai_exec_summary = ask_bedrock(exec_summary_prompt)
                        st.markdown("### 🤖 Bedrock Executive Health Summary")
                        st.info(ai_exec_summary)
                else:
                    st.warning("No ingestion partition telemetry found. Please load days in the 'Daily Load' tab first!")
            except Exception as e:
                st.error(f"Failed to retrieve health metrics: {str(e)}")
