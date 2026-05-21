"""
Sigma DataTech — Multi-Industry Data Generator
Generates customers, products, and 5 days of transaction data.
Day 3 has planted data quality issues for students to discover.
Dates: 2026-05-01 to 2026-05-05

Usage:
    python generate_data.py                  # interactive industry selection
    python generate_data.py --industry 2     # electronics (skip prompt)
"""
import pandas as pd
import numpy as np
import random
import os
import sys
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ── Common reference data ──────────────────────────────────────────────────────
CITIES = [
    'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad',
    'Pune', 'Kolkata', 'Ahmedabad', 'Jaipur', 'Surat'
]
TIERS = ['Gold', 'Silver', 'Bronze']
PAYMENT_METHODS = ['UPI', 'Net Banking', 'Credit Card', 'Debit Card', 'EMI']
STATUSES = ['completed', 'pending', 'failed', 'returned']

FIRST_NAMES = [
    'Rahul', 'Priya', 'Arjun', 'Sneha', 'Vikram', 'Ananya', 'Rohan', 'Kavya',
    'Aditya', 'Divya', 'Kiran', 'Meera', 'Sanjay', 'Neha', 'Amit', 'Pooja',
    'Rajesh', 'Sunita', 'Deepak', 'Anjali', 'Suresh', 'Ritu', 'Manoj', 'Swati',
    'Ajay', 'Nisha', 'Vikas', 'Preeti', 'Nitin', 'Smita', 'Ravi', 'Lakshmi',
    'Ganesh', 'Padma', 'Harish', 'Rekha', 'Mohan', 'Usha', 'Vinod', 'Geeta'
]
LAST_NAMES = [
    'Sharma', 'Gupta', 'Patel', 'Singh', 'Kumar', 'Joshi', 'Rao', 'Nair',
    'Iyer', 'Reddy', 'Shah', 'Mehta', 'Verma', 'Agarwal', 'Mishra', 'Tiwari',
    'Pandey', 'Dubey', 'Jain', 'Malhotra', 'Chopra', 'Bose', 'Das', 'Mukherjee'
]

# ── Industry definitions ───────────────────────────────────────────────────────
INDUSTRIES = {
    1: {
        'name': 'Retail / FMCG',
        'company': 'QuickMart',
        'categories': ['Groceries', 'Personal Care', 'Home Care', 'Beverages', 'Snacks',
                       'Dairy', 'Bakery', 'Frozen Foods', 'Organic', 'Baby Care'],
        'products': [
            'Tata Salt 1kg', 'Aashirvaad Atta 5kg', 'Amul Butter 500g', 'Surf Excel 2kg',
            'Colgate MaxFresh', 'Dove Soap 6pk', 'Maggi Noodles 12pk', 'Lay\'s Chips Variety',
            'Bisleri Water 24pk', 'Red Bull 4pk', 'Amul Gold Milk 1L', 'Britannia Bread',
            'Horlicks 500g', 'Dettol Handwash', 'Pampers Diapers L', 'Good Knight Refill',
            'Parle-G Biscuits', 'Tropicana Juice 1L', 'Nescafe Classic 200g', 'Vim Dish Wash',
            'Vaseline Body Lotion', 'Head & Shoulders', 'Ariel 3kg', 'Lipton Tea 250g',
            'Kissan Jam 500g', 'MDH Masala Kit', 'Everest Garam Masala', 'Fortune Oil 5L',
            'Saffola Gold 5L', 'Himalaya Face Wash', 'Mamaearth Shampoo', 'WOW Apple Cider',
            'Organic India Tulsi Tea', 'Patanjali Ghee 1kg', 'Nestlé KitKat 12pk',
            'Cadbury Dairy Milk', 'Haldiram Namkeen', 'Bikano Rasgulla', 'Amul Kool',
            'Paper Boat Aamras', 'Glucon-D 1kg', 'Complan Chocolate', 'Boost 500g',
            'Pediasure Vanilla', 'Whisper Ultra XL', 'Gillette Mach3', 'Nivea Men',
            'Old Spice Deodorant', 'Park Avenue Perfume', 'Fogg Deodorant'
        ],
        'amount_range': (50, 5000),
        'high_value_threshold': 2000,
    },
    2: {
        'name': 'Electronics',
        'company': 'TechZone',
        'categories': ['Smartphones', 'Laptops', 'Tablets', 'Audio', 'Cameras',
                       'Wearables', 'Gaming', 'Accessories', 'Smart Home', 'Television'],
        'products': [
            'Samsung Galaxy S25', 'iPhone 16 Pro', 'OnePlus 13', 'Pixel 9 Pro', 'Vivo X200',
            'Realme 14 Pro', 'Redmi Note 14 Pro', 'iQOO 13', 'Motorola Edge 50', 'Oppo Reno 13',
            'Dell XPS 15', 'MacBook Air M4', 'HP Spectre x360', 'Lenovo ThinkPad X1', 'Asus ZenBook 14',
            'Acer Swift 5', 'Microsoft Surface Pro 11', 'MSI Prestige 16', 'Razer Blade 15', 'LG Gram 17',
            'iPad Pro M4', 'Samsung Galaxy Tab S10', 'OnePlus Pad 2', 'Lenovo Tab P12 Pro', 'Realme Pad X',
            'Sony WH-1000XM6', 'Bose QC Ultra', 'Apple AirPods Pro 3', 'boAt Airdopes 800', 'Nothing Ear 3',
            'Sony Alpha ZV-E10 II', 'Canon EOS R50', 'GoPro Hero 13', 'DJI Osmo Pocket 3', 'Fujifilm X-T50',
            'Apple Watch Ultra 3', 'Samsung Galaxy Watch 7', 'Garmin Fenix 8', 'Noise ColorFit Ultra', 'Fitbit Charge 7',
            'PlayStation 5 Slim', 'Xbox Series X', 'Nintendo Switch 2', 'Razer DeathAdder V3', 'Logitech G Pro X',
            'Amazon Echo Show 15', 'Google Nest Hub Max', 'Ring Video Doorbell Pro', 'Philips Hue Kit',
            'Samsung QLED 65"', 'LG OLED C4 55"'
        ],
        'amount_range': (999, 200000),
        'high_value_threshold': 50000,
    },
    3: {
        'name': 'Logistics / Supply Chain',
        'company': 'FleetTrack',
        'categories': ['Express Delivery', 'Freight', 'Cold Chain', 'Bulk Cargo',
                       'Last Mile', 'Reverse Logistics', 'Cross Border', 'Warehousing',
                       'Same Day', 'Scheduled'],
        'products': [
            'Express Parcel 0.5kg', 'Express Parcel 2kg', 'Express Parcel 5kg',
            'Freight LTL 50kg', 'Freight LTL 100kg', 'Freight FTL 500kg',
            'Cold Chain Pharma', 'Cold Chain Food', 'Cold Chain Chemicals',
            'Bulk Cement 1MT', 'Bulk Steel 5MT', 'Bulk Grain 2MT',
            'Last Mile Urban', 'Last Mile Rural', 'Last Mile Express',
            'Reverse Pickup Standard', 'Reverse Pickup Express',
            'Cross Border India-UAE', 'Cross Border India-US', 'Cross Border India-UK',
            'Warehouse Storage Monthly', 'Warehouse Storage Weekly',
            'Same Day Metro', 'Same Day Tier2', 'Scheduled Weekly',
            'Scheduled Bi-Weekly', 'Document Courier', 'Fragile Items Express',
            'Hazardous Cargo Certified', 'Oversized Cargo Special',
            'E-commerce Fulfillment Basic', 'E-commerce Fulfillment Premium',
            'Pharmaceutical GDP Certified', 'Temperature Controlled -18C',
            'Temperature Controlled 2-8C', 'White Glove Delivery',
            'Installation Service', 'Assembly Service', 'Packaging Service',
            'Insurance Add-on Basic', 'Insurance Add-on Premium',
            'Track and Trace Basic', 'Track and Trace Advanced',
            'Customs Clearance Standard', 'Customs Clearance Express',
            'Air Freight Economy', 'Air Freight Priority',
            'Sea Freight 20ft Container', 'Sea Freight 40ft Container',
            'Rail Freight Standard', 'Drone Delivery Pilot'
        ],
        'amount_range': (200, 500000),
        'high_value_threshold': 100000,
    },
    4: {
        'name': 'Healthcare',
        'company': 'MediCare',
        'categories': ['Consultation', 'Diagnostics', 'Pharmacy', 'Surgery',
                       'Physiotherapy', 'Dental', 'Ophthalmology', 'Cardiology',
                       'Radiology', 'Emergency'],
        'products': [
            'General Consultation', 'Specialist Consultation', 'Teleconsultation',
            'CBC Blood Test', 'Lipid Profile', 'Thyroid Panel', 'HbA1c Test',
            'X-Ray Chest', 'MRI Brain', 'CT Scan Abdomen', 'Ultrasound Pelvis',
            'ECG Standard', 'Echo Cardiogram', 'Stress Test',
            'Pharmacy Pack Basic', 'Pharmacy Pack Chronic', 'Pharmacy Pack Acute',
            'Minor Surgery OPD', 'Day Care Surgery', 'Major Surgery',
            'Physiotherapy Session', 'Physiotherapy 10 Pack', 'Physiotherapy Monthly',
            'Dental Cleaning', 'Dental Filling', 'Root Canal', 'Dental Crown',
            'Eye Checkup', 'Glasses Prescription', 'LASIK Consultation',
            'Cardiology Review', 'Angiography', 'Angioplasty',
            'Mammography', 'Bone Density Scan', 'PET CT Scan',
            'Vaccination Basic', 'Vaccination Travel', 'Health Checkup Basic',
            'Health Checkup Comprehensive', 'Health Checkup Executive',
            'ICU Per Day', 'Room Charges General', 'Room Charges Private',
            'Ambulance Basic', 'Ambulance Advanced Life Support',
            'Home Nursing Per Day', 'Home Sample Collection',
            'Mental Health Consultation', 'Nutrition Consultation', 'Dietitian Package'
        ],
        'amount_range': (200, 200000),
        'high_value_threshold': 50000,
    },
    5: {
        'name': 'EdTech',
        'company': 'LearnArc',
        'categories': ['School K-12', 'College Prep', 'Professional Certs', 'Coding',
                       'Data Science', 'MBA Prep', 'Language Learning', 'Creative Arts',
                       'Test Prep', 'Corporate Training'],
        'products': [
            'Class 10 Math Annual', 'Class 12 Science Pack', 'IIT-JEE Foundation',
            'NEET Preparation 1Y', 'JEE Main Crash Course', 'NEET Crash Course',
            'CA Foundation Course', 'CS Executive Course', 'CMA Inter Course',
            'Python for Beginners', 'Full Stack Web Dev', 'Data Science Bootcamp',
            'Machine Learning Pro', 'Cloud AWS Certification', 'DevOps Bootcamp',
            'MBA CAT Preparation', 'MBA GMAT Course', 'MBA Interview Prep',
            'IELTS 30-Day Course', 'TOEFL Preparation', 'Spoken English 3M',
            'French Beginner', 'Spanish Intermediate', 'German Advanced',
            'Digital Photography', 'UI/UX Design Course', 'Video Editing Pro',
            'Music Production', 'Graphic Design Adobe', 'Creative Writing',
            'GRE Verbal Math', 'UPSC Prelims Course', 'SSC CGL Preparation',
            'Banking PO Course', 'RBI Grade B Prep', 'Insurance IRDA Cert',
            'Excel Advanced', 'Power BI Course', 'Tableau Certification',
            'Project Management PMP', 'Agile Scrum Master', 'Six Sigma Green Belt',
            'Leadership Programme', 'Communication Skills', 'Presentation Skills',
            'AI for Business', 'GenAI Foundations', 'Prompt Engineering',
            'Cybersecurity Basics', 'Ethical Hacking', 'Network Security'
        ],
        'amount_range': (999, 150000),
        'high_value_threshold': 30000,
    },
    6: {
        'name': 'Hospitality / Hotels',
        'company': 'StayElite',
        'categories': ['Room Booking', 'F&B', 'Spa & Wellness', 'Events',
                       'Business Services', 'Transport', 'Tours', 'Packages',
                       'Memberships', 'Catering'],
        'products': [
            'Standard Room 1N', 'Deluxe Room 1N', 'Suite 1N', 'Presidential Suite 1N',
            'Standard Room 3N', 'Deluxe Room 3N', 'Suite 3N',
            'Standard Room 7N', 'Deluxe Room 7N', 'Suite 7N',
            'Breakfast Buffet', 'Lunch Buffet', 'Dinner Buffet', 'High Tea',
            'A La Carte Dinner', 'Pool Bar Package', 'Mini Bar Restock',
            'Swedish Massage 60min', 'Deep Tissue 90min', 'Couple Spa Package',
            'Jacuzzi Access', 'Steam & Sauna', 'Yoga Session',
            'Conference Hall Half Day', 'Conference Hall Full Day', 'Boardroom 4hr',
            'Wedding Package Basic', 'Wedding Package Premium', 'Wedding Package Luxury',
            'Airport Pickup Sedan', 'Airport Pickup SUV', 'Limousine Transfer',
            'City Tour Half Day', 'City Tour Full Day', 'Heritage Walk',
            'Weekend Getaway Package', 'Honeymoon Package', 'Family Vacation Package',
            'Annual Membership Silver', 'Annual Membership Gold', 'Annual Membership Platinum',
            'Corporate Catering 50pax', 'Corporate Catering 100pax', 'Box Lunch 20pax',
            'Pool Access Day Pass', 'Gym Day Pass', 'Childcare Service',
            'Late Checkout Fee', 'Early Checkin Fee', 'Laundry Express'
        ],
        'amount_range': (500, 500000),
        'high_value_threshold': 50000,
    },
    7: {
        'name': 'Fintech / Banking',
        'company': 'SigmaPay',
        'categories': ['Payments', 'Lending', 'Insurance', 'Investments',
                       'Transfers', 'Cards', 'Savings', 'Business', 'Crypto', 'NRI'],
        'products': [
            'SigmaPay Basic', 'SigmaPay Pro', 'SigmaPay Business',
            'QuickLoan 50K', 'QuickLoan 2L', 'QuickLoan 5L', 'QuickLoan 10L',
            'Term Insurance Plus', 'Health Shield', 'Motor Cover', 'Travel Insurance',
            'SIP Starter 1K', 'SIP Pro 5K', 'MutualFund Direct', 'Index Fund',
            'Gold Savings Digital', 'Fixed Deposit 1Y', 'Fixed Deposit 3Y',
            'UPI Turbo', 'Bharat QR Pro', 'International UPI',
            'Credit Card Rewards', 'Credit Card Cashback', 'Debit Card Premium',
            'Forex Card', 'Prepaid Travel Card',
            'PayLater 30', 'PayLater 90', 'BNPL Lite',
            'RD Monthly 2K', 'RD Monthly 5K', 'Savings Account Pro',
            'Current Account Business', 'Payroll API', 'Collections Suite',
            'Remittance Plus', 'NRI Account', 'NRI Fixed Deposit',
            'Personal Loan 1L', 'Home Loan Assist', 'Car Loan Fast',
            'Education Loan', 'Supply Chain Finance',
            'Bitcoin Wallet', 'Crypto SIP', 'Stablecoin Savings',
            'CreditBoost', 'ScoreTracker', 'NPS Lite', 'ELSS Fund'
        ],
        'amount_range': (100, 1000000),
        'high_value_threshold': 100000,
    },
    8: {
        'name': 'E-commerce / Fashion',
        'company': 'TrendCart',
        'categories': ['Men\'s Fashion', 'Women\'s Fashion', 'Kids', 'Footwear',
                       'Accessories', 'Sports', 'Beauty', 'Home Decor', 'Jewellery', 'Ethnic Wear'],
        'products': [
            'Levi\'s 511 Jeans', 'Zara Slim Fit Shirt', 'H&M Casual Tee 3pk',
            'Allen Solly Formal', 'Van Heusen Blazer', 'Arrow Formal Set',
            'Saree Silk Banarasi', 'Kurti Set Premium', 'Anarkali Embroidered',
            'Lehenga Bridal', 'Salwar Kameez Set', 'Western Top Denim',
            'Kids Ethnic Set', 'Kids School Set', 'Kids Casuals 3pk',
            'Nike Air Max 2026', 'Adidas Ultraboost', 'Puma RS-X',
            'Red Tape Formal', 'Bata Oxford', 'Metro Heels',
            'Fastrack Watch', 'Titan Raga', 'Casio G-Shock',
            'Ray-Ban Aviator', 'Fossil Smartwatch', 'Luxury Bag Replica',
            'Nike Dri-FIT Set', 'Puma Training Kit', 'Adidas Yoga Mat',
            'Lakme Foundation', 'Maybelline Kit', 'L\'Oreal Serum',
            'Forest Essentials', 'Kama Ayurveda Set', 'Biotique Pack',
            'Curtains Premium', 'Bedsheet King Size', 'Cushion Cover Set',
            'Gold Necklace 22K', 'Diamond Earrings', 'Silver Bracelet',
            'Kundan Set Bridal', 'Temple Jewellery', 'Oxidised Set',
            'Banarasi Silk Dupatta', 'Phulkari Suit', 'Chanderi Saree',
            'Denim Jacket Unisex', 'Hoodie Premium', 'Athleisure Set'
        ],
        'amount_range': (200, 100000),
        'high_value_threshold': 10000,
    },
}


def select_industry() -> dict:
    """Interactive industry selection or via command line argument."""
    # Check for --industry flag
    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--industry' and i + 1 < len(sys.argv) - 1:
            try:
                choice = int(sys.argv[i + 2])
                if choice in INDUSTRIES:
                    print(f"✅ Industry selected: {INDUSTRIES[choice]['name']}")
                    return INDUSTRIES[choice]
            except (ValueError, IndexError):
                pass

    print("\n" + "="*50)
    print("  SIGMA DATATECH — Industry Selector")
    print("="*50)
    for key, val in INDUSTRIES.items():
        print(f"  {key}. {val['name']} ({val['company']})")
    print("="*50)

    while True:
        try:
            choice = int(input("\nSelect industry (1-8): ").strip())
            if choice in INDUSTRIES:
                print(f"\n✅ Selected: {INDUSTRIES[choice]['name']} — {INDUSTRIES[choice]['company']}")
                return INDUSTRIES[choice]
            else:
                print("Invalid choice. Enter a number between 1 and 8.")
        except ValueError:
            print("Please enter a valid number.")


def generate_customers(n=200):
    customers = []
    for i in range(n):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        email = f"{first.lower()}.{last.lower()}{i}@gmail.com"
        phone = f"+91{random.randint(7000000000, 9999999999)}"
        tier = random.choices(TIERS, weights=[0.20, 0.35, 0.45])[0]
        signup = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 500))
        customers.append({
            'customer_id': f'CUST{str(i + 1).zfill(4)}',
            'name': f'{first} {last}',
            'email': email,
            'phone': phone,
            'city': random.choice(CITIES),
            'tier': tier,
            'signup_date': signup.strftime('%Y-%m-%d'),
        })
    return pd.DataFrame(customers)


def generate_products(industry: dict):
    products = []
    product_names = industry['products']
    categories = industry['categories']
    low, high = industry['amount_range']

    for i, name in enumerate(product_names):
        category = categories[i % len(categories)]
        price = round(random.uniform(low, min(high, low * 50)), 2)
        products.append({
            'product_id': f'PROD{str(i + 1).zfill(3)}',
            'name': name,
            'category': category,
            'price': price,
            'stock_quantity': random.randint(10, 500),
            'is_active': random.choices([True, False], weights=[0.92, 0.08])[0],
        })
    return pd.DataFrame(products)


def generate_orders(industry: dict, day_num: int, date_str: str, n=200, plant_issues=False):
    customer_ids = [f'CUST{str(i + 1).zfill(4)}' for i in range(200)]
    product_ids = [f'PROD{str(i + 1).zfill(3)}' for i in range(len(industry['products']))]
    low, high = industry['amount_range']
    base = (day_num - 1) * n + 1

    rows = []
    for i in range(n):
        hour = random.randint(8, 23)
        minute = random.randint(0, 59)
        quantity = random.choices([1, 2, 3], weights=[0.80, 0.15, 0.05])[0]
        amount = round(random.uniform(low, high) * quantity, 2)

        rows.append({
            'order_id': f'ORD{str(base + i).zfill(6)}',
            'customer_id': random.choice(customer_ids),
            'product_id': random.choice(product_ids),
            'quantity': quantity,
            'amount': amount,
            'status': random.choices(STATUSES, weights=[0.78, 0.10, 0.07, 0.05])[0],
            'payment_method': random.choice(PAYMENT_METHODS),
            'city': random.choice(CITIES),
            'created_at': f"{date_str} {str(hour).zfill(2)}:{str(minute).zfill(2)}:00",
        })

    df = pd.DataFrame(rows)

    if plant_issues:
        # Issue 1: 10 negative amounts (refund misposted as new order)
        for idx in random.sample(range(n), 10):
            df.at[idx, 'amount'] = -abs(df.at[idx, 'amount'])

        # Issue 2: 5 duplicate order_ids (payment retry without idempotency)
        for k in range(5):
            df.at[n - k - 1, 'order_id'] = df.at[k, 'order_id']

        # Issue 3: 3 null customer_ids (guest checkout — linkage failed)
        for idx in random.sample(range(n), 3):
            df.at[idx, 'customer_id'] = None

        # Issue 4: 4 zero-quantity orders (cart submission bug)
        for idx in random.sample(range(n), 4):
            df.at[idx, 'quantity'] = 0

    return df


if __name__ == '__main__':
    industry = select_industry()
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"\n📦 Generating data for: {industry['name']} — {industry['company']}")
    print("-" * 50)

    customers = generate_customers(200)
    customers.to_csv(os.path.join(out_dir, 'customers.csv'), index=False)
    print(f"✅ customers.csv     — {len(customers)} rows")

    products = generate_products(industry)
    products.to_csv(os.path.join(out_dir, 'products.csv'), index=False)
    print(f"✅ products.csv      — {len(products)} rows  ({industry['name']})")

    dates = ['2026-05-01', '2026-05-02', '2026-05-03', '2026-05-04', '2026-05-05']
    for day_num, date_str in enumerate(dates, start=1):
        has_issues = (day_num == 3)
        orders = generate_orders(industry, day_num, date_str, n=200, plant_issues=has_issues)
        fname = os.path.join(out_dir, f'orders_day{day_num}.csv')
        orders.to_csv(fname, index=False)
        tag = '⚠️  (10 neg amounts, 5 dup IDs, 3 null customers, 4 zero-qty PLANTED)' if has_issues else ''
        print(f"✅ orders_day{day_num}.csv  — {len(orders)} rows  {tag}")

    print(f"\n✅ All data ready — {industry['company']} | {industry['name']} | May 2026")
