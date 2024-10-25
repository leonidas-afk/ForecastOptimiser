const express = require("express");
const fs = require("fs");
const csv = require("csv-parser");
const path = require("path");
const sqlite3 = require('sqlite3').verbose();
const dbPath = './data/measurements.db';


const app = express();
app.use(express.json());

const port = 3000;

const db = new sqlite3.Database(dbPath, (err) => {
  if (err) {
    console.error(err.message);
  }
  console.log('Connected to the database.');
});

app.post("/get-data", (req, res) => {
  const { startDate, endDate, table } = req.body;

  // Query the database
  const query = `SELECT * FROM ${table} WHERE datetime >= ? AND datetime <= ?`;

  db.all(query, [startDate, endDate], (err, rows) => {
    if (err) {
      return res.status(500).json({ error: err.message });
    }
    return res.json({ data: rows });
  });
});

app.get("/koko", (req, res) => {
  return res.json({ hello: "O2K" });
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
