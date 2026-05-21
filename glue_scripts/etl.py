import boto3
import pandas as pd
import json
import sys
import datetime
import io
import logging

# Configure logging to standard output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing AWS Glue Python Shell ETL script...")
    
    try:
        # Parse arguments manually from sys.argv to adhere strictly to allowed imports
        logger.info("Parsing command line arguments...")
        args = {}
        for i in range(1, len(sys.argv)):
            arg = sys.argv[i]
            if arg.startswith('--'):
                name = arg[2:]
                # Next token is the value (if exists and doesn't start with --)
                if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
                    args[name] = sys.argv[i + 1]
                else:
                    args[name] = True
        
        bucket_name = args.get('bucket_name')
        date_partition = args.get('date_partition')
        job_type = args.get('job_type')
        
        if not bucket_name or not date_partition or not job_type:
            missing_args = [k for k in ['bucket_name', 'date_partition', 'job_type'] if not args.get(k)]
            error_msg = f"Missing required arguments: {', '.join('--' + x for x in missing_args)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info(f"Arguments parsed successfully. job_type: {job_type}, bucket_name: {bucket_name}, date_partition: {date_partition}")
        
        s3_client = boto3.client('s3')
        
        if job_type == "orders":
            logger.info("Processing job_type: orders")
            
            raw_key = f"raw/orders/date={date_partition}/orders.csv"
            processed_key = f"processed/orders/date={date_partition}/orders.csv"
            report_key = f"reports/quality_report_{date_partition}.json"
            
            # Read from raw S3 CSV
            logger.info(f"Reading raw orders CSV from s3://{bucket_name}/{raw_key}")
            try:
                response = s3_client.get_object(Bucket=bucket_name, Key=raw_key)
                csv_data = response['Body'].read()
            except Exception as e:
                logger.error(f"Error reading orders CSV from S3: {str(e)}")
                raise
                
            df = pd.read_csv(io.BytesIO(csv_data))
            input_rows = len(df)
            logger.info(f"Read {input_rows} rows from raw orders.")
            
            # Count and log quality issues before fixing
            null_customer_ids = int(df['customer_id'].isnull().sum())
            
            # Coerce amount to numeric to safely handle inequalities
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            negative_amounts = int((df['amount'] < 0).sum())
            
            duplicate_order_ids = int(df.duplicated(subset=['order_id']).sum())
            
            logger.info(f"Quality issues identified:")
            logger.info(f"  - Null customer_ids: {null_customer_ids}")
            logger.info(f"  - Negative amounts: {negative_amounts}")
            logger.info(f"  - Duplicate order_ids: {duplicate_order_ids}")
            
            # Fixes
            logger.info("Applying data quality fixes...")
            
            # 1. drop null customer_ids
            df = df.dropna(subset=['customer_id'])
            
            # 2. abs() negative amounts
            df['amount'] = df['amount'].abs()
            
            # 3. drop_duplicates on order_id keeping first
            df = df.drop_duplicates(subset=['order_id'], keep='first')
            
            output_rows = len(df)
            rows_dropped = input_rows - output_rows
            logger.info(f"Data quality fixes applied. Output rows: {output_rows}, Rows dropped: {rows_dropped}")
            
            # Add columns
            # Add column processed_at = current UTC timestamp (string)
            current_utc = datetime.datetime.now(datetime.timezone.utc)
            processed_at_str = current_utc.strftime('%Y-%m-%d %H:%M:%S')
            df['processed_at'] = processed_at_str
            
            # Add column is_high_value = True if amount > 10000 else False
            df['is_high_value'] = (df['amount'] > 10000).astype(str)
            
            # Write cleaned CSV to s3://{bucket_name}/processed/orders/date={date_partition}/orders.csv
            logger.info(f"Writing cleaned CSV to s3://{bucket_name}/{processed_key}")
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            
            try:
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=processed_key,
                    Body=csv_buffer.getvalue()
                )
                logger.info("Successfully wrote cleaned CSV to S3.")
            except Exception as e:
                logger.error(f"Error writing cleaned CSV to S3: {str(e)}")
                raise
                
            # Write JSON quality report to s3://{bucket_name}/reports/quality_report_{date_partition}.json
            logger.info(f"Writing quality report JSON to s3://{bucket_name}/{report_key}")
            report = {
                "date": date_partition,
                "input_rows": input_rows,
                "output_rows": output_rows,
                "null_customer_ids": null_customer_ids,
                "negative_amounts": negative_amounts,
                "duplicate_order_ids": duplicate_order_ids,
                "rows_dropped": rows_dropped,
                "status": "SUCCESS"
            }
            
            try:
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=report_key,
                    Body=json.dumps(report, indent=4)
                )
                logger.info("Successfully wrote quality report JSON to S3.")
            except Exception as e:
                logger.error(f"Error writing quality report to S3: {str(e)}")
                raise
                
        elif job_type == "reference":
            logger.info("Processing job_type: reference")
            
            for file_name in ["customers.csv", "products.csv"]:
                src_key = f"raw/{file_name}"
                dst_key = f"processed/{file_name}"
                
                logger.info(f"Copying {file_name} from s3://{bucket_name}/{src_key} to s3://{bucket_name}/{dst_key} unchanged")
                try:
                    s3_client.copy_object(
                        Bucket=bucket_name,
                        CopySource={'Bucket': bucket_name, 'Key': src_key},
                        Key=dst_key
                    )
                    logger.info(f"Successfully copied {file_name}")
                except Exception as e:
                    logger.error(f"Error copying reference file {file_name}: {str(e)}")
                    raise
                    
            logger.info("Reference file copy complete.")
            
        else:
            error_msg = f"Unknown job_type: {job_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info("ETL Script execution completed successfully.")
        
    except Exception as e:
        logger.critical(f"ETL Script failed with exception: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
