#!/bin/bash

# Define the SQLite database name
DB_NAME="measurements.db"

# Iterate over all CSV files in the current directory
for csv_file in *.csv; do
    # Extract the table name from the file name (remove the .csv extension)
    table_name=$(basename "$csv_file" .csv)
    
    # Create the table in the SQLite database with the specified structure
    sqlite3 "$DB_NAME" <<EOF
DROP TABLE IF EXISTS $table_name;
CREATE TABLE IF NOT EXISTS $table_name (
    measurement_date DATETIME,
    value DOUBLE
);
EOF

    # Import the data from the CSV file into the SQLite table
    sqlite3 "$DB_NAME" <<EOF
.mode csv
.import $csv_file $table_name
EOF

    # Fix the headers
    sqlite3 "$DB_NAME" <<EOF
DELETE FROM $table_name WHERE measurement_date = 'measurement_date';
EOF

done

echo "Data import complete."
