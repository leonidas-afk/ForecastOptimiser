#!/bin/bash

# Function to insert data from a CSV file into an SQLite database table
insert_data() {
    local CSV_FILE=$1
    local DB_NAME=$2
    local TABLE_NAME=$3

    # Create the SQLite database and table
    sqlite3 $DB_NAME << EOF
DROP TABLE IF EXISTS $TABLE_NAME;
EOF
    sqlite3 $DB_NAME <<EOF
CREATE TABLE IF NOT EXISTS $TABLE_NAME (
    datetime TEXT,
    value REAL
);
EOF

    # Insert data from CSV into the table
    sqlite3 $DB_NAME <<EOF
.mode csv
.import $CSV_FILE $TABLE_NAME
EOF

    echo "Data has been inserted into $TABLE_NAME in $DB_NAME"
}

# Example of how to call the function
# insert_data "data.csv" "lstm_data.db" "LSTM1h"