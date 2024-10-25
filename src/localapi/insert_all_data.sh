#!/bin/bash

source ./insert_data.sh

insert_data "data/LSTM15m.csv" "data/measurements.db" "LSTM15m"
insert_data "data/LSTM30m.csv" "data/measurements.db" "LSTM30m"
insert_data "data/LSTM1h.csv" "data/measurements.db" "LSTM1h"
insert_data "data/LSTM2h.csv" "data/measurements.db" "LSTM2h"
insert_data "data/LSTM4h.csv" "data/measurements.db" "LSTM4h"
insert_data "data/LSTM8h.csv" "data/measurements.db" "LSTM8h"
insert_data "data/LSTM1d.csv" "data/measurements.db" "LSTM1d"
insert_data "data/LSTM1w.csv" "data/measurements.db" "LSTM1w"
insert_data "data/LSTM1M.csv" "data/measurements.db" "LSTM1M"


