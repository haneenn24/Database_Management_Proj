#Example of Input Query - house pricing Database:

### ✅ 1. **Selection + Conjunctive Query**

> **Query:**  
Select houses where `LotArea > 8000` **AND** `SalePrice < 200000`.

```sql
SELECT * FROM house_prices
WHERE LotArea > 8000 AND SalePrice < 200000;
```

📌 Filters rows using **multiple conditions** (AND).

---

### ✅ 2. **Selection + Union of Conjunctive Queries (UCQ)**

> **Query:**  
Select houses that are either:
- In `Neighborhood = 'CollgCr'` AND `SalePrice > 250000`,  
**OR**
- In `Neighborhood = 'OldTown'` AND `YearBuilt > 2000`

```sql
SELECT * FROM house_prices
WHERE (Neighborhood = 'CollgCr' AND SalePrice > 250000)
   OR (Neighborhood = 'OldTown' AND YearBuilt > 2000);
```

📌 Models **uncertain logical alternatives**.

---

### ✅ 3. **Projection + Conjunctive Query**

> **Query:**  
Get only the `LotArea` and `SalePrice` where `OverallQual >= 7`.

```sql
SELECT LotArea, SalePrice FROM house_prices
WHERE OverallQual >= 7;
```

📌 Returns **specific columns**, not full rows.

---

### ✅ 4. **Projection + UCQ**

> **Query:**  
Project `GrLivArea`, `SalePrice` for homes with:
- `HouseStyle = '1Story'` OR  
- `HouseStyle = '2Story'` AND `SalePrice > 300000`

```sql
SELECT GrLivArea, SalePrice FROM house_prices
WHERE (HouseStyle = '1Story')
   OR (HouseStyle = '2Story' AND SalePrice > 300000);
```

---

### ✅ 5. **Join + Conjunctive Query**

> **Assume another table:** `owners(PID, Name, Neighborhood)`

> **Query:**  
Join with owners table to find homes in `NAmes` with `GrLivArea > 1500`.

```sql
SELECT hp.*, o.Name
FROM house_prices hp
JOIN owners o ON hp.Neighborhood = o.Neighborhood
WHERE hp.GrLivArea > 1500 AND o.Neighborhood = 'NAmes';
```

📌 Combines structured data across tables (important in real-world use).

---

### ✅ 6. **Join + UCQ**

> **Query:**  
Find homes with either:
- `GarageCars >= 2` in `CollgCr`,  
OR  
- `SalePrice > 300000` in `Edwards`

```sql
SELECT hp.*, o.Name
FROM house_prices hp
JOIN owners o ON hp.Neighborhood = o.Neighborhood
WHERE (hp.GarageCars >= 2 AND hp.Neighborhood = 'CollgCr')
   OR (hp.SalePrice > 300000 AND hp.Neighborhood = 'Edwards');
```

---

### ✅ 7. **Union + Conjunctive Query**

> **Query:**  
Union two result sets:
- Houses with `YearBuilt < 1970 AND SalePrice < 150000`
- Houses with `LotArea > 10000 AND OverallQual >= 8`

```sql
SELECT * FROM house_prices
WHERE YearBuilt < 1970 AND SalePrice < 150000
UNION
SELECT * FROM house_prices
WHERE LotArea > 10000 AND OverallQual >= 8;
```

---

### ✅ 8. **Union + UCQ**

> **Query:**  
Union of two UCQs:
- `(Neighborhood = 'NAmes' AND SalePrice < 120000)`  
- `(Neighborhood = 'BrkSide' AND GrLivArea > 2000)`

```sql
SELECT * FROM house_prices
WHERE Neighborhood = 'NAmes' AND SalePrice < 120000
UNION
SELECT * FROM house_prices
WHERE Neighborhood = 'BrkSide' AND GrLivArea > 2000;
```

✅ SAFE queries:

Simple selections with no joins

Projections on independent attributes

❌ #P-complete (Unsafe) queries:

Projections involving dependent attributes (e.g., LotArea ←→ SalePrice)

Joins with shared, correlated variables like Neighborhood or SalePrice








