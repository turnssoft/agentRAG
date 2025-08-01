# inventory_manager.py
# Author: Tony Pepperoni
# Date: 2025-07-15
# Description: A script to manage and report on daily inventory levels for Pizza Boys.

import smtplib
from email.mime.text import MIMEText
from datetime import datetime

# --- Configuration ---
SMTP_SERVER = "smtp.pizzaboys.internal"
SMTP_PORT = 587
ALERT_RECIPIENTS = ["manager@pizzaboys.com", "kitchen_lead@pizzaboys.com"]
FROM_EMAIL = "inventory-bot@pizzaboys.com"

# --- Data Store ---
# This dictionary represents our live inventory database.
INVENTORY = {
    "dough_ball_std": {"name": "Standard Dough Ball", "current_kg": 80, "min_kg": 100},
    "dough_ball_gf": {"name": "Gluten-Free Dough Ball", "current_kg": 15, "min_kg": 20},
    "sauce_classic": {"name": "Classic Tomato Sauce", "current_liters": 50, "min_liters": 40},
    "sauce_nduja": {"name": "Spicy 'Nduja Sauce", "current_liters": 10, "min_liters": 15},
    "cheese_mozz": {"name": "Shredded Mozzarella", "current_kg": 60, "min_kg": 50},
    "pepp_cup": {"name": "Cup-and-Char Pepperoni", "current_kg": 25, "min_kg": 30},
    "sausage_spicy": {"name": "Spicy Italian Sausage", "current_kg": 18, "min_kg": 20},
    "jalapenos_fresh": {"name": "Fresh Jalapeños", "current_kg": 5, "min_kg": 8},
}

def send_alert(ingredient_name, current_level, min_level, unit):
    """
    Sends an email alert for a specific low-stock ingredient.
    
    Args:
        ingredient_name (str): The name of the ingredient.
        current_level (float): The current stock level.
        min_level (float): The minimum required stock level.
        unit (str): The unit of measurement (e.g., 'kg', 'liters').
    """
    subject = f"URGENT: Low Inventory Alert for {ingredient_name}"
    body = f"""
    This is an automated alert from the Pizza Boys Inventory Management System.
    
    The stock level for '{ingredient_name}' is critically low.
    
    Current Level: {current_level} {unit}
    Minimum Required Level: {min_level} {unit}
    
    Please place an order with our supplier immediately to avoid a service disruption.
    """
    
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = FROM_EMAIL
    msg['To'] = ", ".join(ALERT_RECIPIENTS)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            # server.login(USERNAME, PASSWORD) # Authentication might be needed
            server.sendmail(FROM_EMAIL, ALERT_RECIPIENTS, msg.as_string())
        print(f"INFO: Successfully sent alert for {ingredient_name}.")
    except Exception as e:
        # Log the error but don't crash the script. The report is still valuable.
        print(f"ERROR: Could not send email alert for {ingredient_name}. Reason: {e}")
        print("INFO: The alert content was:\n---")
        print(f"To: {msg['To']}\nSubject: {subject}\n{body}\n---")


def check_inventory_levels():
    """
    Checks all inventory items and triggers alerts for any below the minimum level.
    Returns a list of items that are low on stock.
    """
    print("Running daily inventory check...")
    low_stock_items = []
    
    for item_id, details in INVENTORY.items():
        unit = ""
        current_level = 0
        min_level = 0

        if "current_kg" in details:
            unit = "kg"
            current_level = details["current_kg"]
            min_level = details["min_kg"]
        elif "current_liters" in details:
            unit = "liters"
            current_level = details["current_liters"]
            min_level = details["min_liters"]
        
        if current_level < min_level:
            print(f"WARNING: {details['name']} is below minimum stock level.")
            low_stock_items.append(details['name'])
            send_alert(details['name'], current_level, min_level, unit)
            
    if not low_stock_items:
        print("All inventory levels are sufficient.")
        
    return low_stock_items

def generate_daily_report():
    """
    Generates and prints a full inventory status report to the console.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"\n--- Pizza Boys Daily Inventory Report ---\n"
    report += f"Generated on: {timestamp}\n\n"
    report += f"{'Ingredient':<30} | {'Current Stock':<15} | {'Minimum Stock':<15} | {'Status':<10}\n"
    report += "-" * 80 + "\n"

    for item_id, details in INVENTORY.items():
        if "current_kg" in details:
            unit = "kg"
            current = details["current_kg"]
            minimum = details["min_kg"]
        else:
            unit = "liters"
            current = details["current_liters"]
            minimum = details["min_liters"]

        status = "OK" if current >= minimum else "LOW"
        report += f"{details['name']:<30} | {f'{current} {unit}':<15} | {f'{minimum} {unit}':<15} | {status:<10}\n"
        
    report += "-" * 80 + "\n"
    print(report)
    return report

if __name__ == "__main__":
    # This block runs when the script is executed directly.
    print("Pizza Boys Automated Inventory System Initializing...")
    
    # 1. Check all inventory levels and send alerts if necessary.
    check_inventory_levels()
    
    # 2. Generate the full daily report for the records.
    generate_daily_report()
    
    print("Inventory check complete.")

