
-- Q1: Customers whose completed orders above INR 500
-- have a combined value above INR 8,000
SELECT customer_id, SUM(order_amount) AS total_spend
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;


-- Q2: Revenue by city and product category
SELECT c.city, p.category,
       ROUND(SUM(o.order_amount), 2) AS revenue_inr
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;


-- Q3: Products priced above the average catalogue price
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (
    SELECT AVG(price_inr) FROM products
)
ORDER BY price_inr DESC;


-- Q4: Customers who have no completed order
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o
    ON c.customer_id = o.customer_id
    AND o.status = 'completed'
WHERE o.order_id IS NULL;


-- Q5: Customers whose completed spend is above
-- the average completed spend per customer
WITH customer_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_spend AS (
    SELECT AVG(total_spend) AS avg_spend
    FROM customer_spend
)
SELECT cs.customer_id, cs.total_spend
FROM customer_spend cs
CROSS JOIN average_spend a
WHERE cs.total_spend > a.avg_spend;


-- Q6: Ranking completed spend within each city
WITH customer_spend AS (
    SELECT c.city, o.customer_id,
           SUM(o.order_amount) AS total_spend
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, o.customer_id
)
SELECT city, customer_id, total_spend,
       RANK() OVER (
           PARTITION BY city ORDER BY total_spend DESC
       ) AS spend_rank,
       DENSE_RANK() OVER (
           PARTITION BY city ORDER BY total_spend DESC
       ) AS dense_rank,
       ROW_NUMBER() OVER (
           PARTITION BY city ORDER BY total_spend DESC
       ) AS row_num
FROM customer_spend
ORDER BY city, spend_rank, customer_id;


-- Q7: Daily completed revenue and running total
WITH daily_revenue AS (
    SELECT substr(order_ts, 1, 10) AS day,
           SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY substr(order_ts, 1, 10)
)
SELECT day,
       ROUND(revenue, 2) AS revenue_inr,
       ROUND(
           SUM(revenue) OVER (ORDER BY day), 2
       ) AS cumulative_revenue_inr
FROM daily_revenue
ORDER BY day;
