import sqlite3

DB = "donors.db"

def update_status():

    donor_id = int(input("Enter Donor ID: "))
    status = input("Available? (y/n): ").lower()

    value = 1 if status == "y" else 0

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE donors SET available=? WHERE donor_id=?",
        (value, donor_id)
    )

    conn.commit()
    conn.close()

    print("✅ Donor status updated.")


def view_donors():

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    rows = cursor.execute(
        "SELECT donor_id,name,blood_group,location,available FROM donors"
    ).fetchall()

    conn.close()

    for r in rows:
        print(r)


if __name__ == "__main__":

    print("\nLife-Link Admin Console")
    print("1. View donors")
    print("2. Update availability")

    choice = input("Select option: ")

    if choice == "1":
        view_donors()
    elif choice == "2":
        update_status()