-- A few analytical queries to try in the shell:
--   prism data/employees.csv data/departments.csv
-- then paste any of these (they end with a semicolon).

-- Headcount and pay band per department.
SELECT department,
       COUNT(*)              AS headcount,
       MIN(salary)           AS lowest,
       MAX(salary)           AS highest,
       ROUND(AVG(salary), 0) AS average
FROM employees
WHERE salary IS NOT NULL
GROUP BY department
ORDER BY average DESC;

-- Only the departments with more than one person, ranked by average pay.
SELECT department, ROUND(AVG(salary), 0) AS avg_salary
FROM employees
GROUP BY department
HAVING COUNT(*) > 1
ORDER BY avg_salary DESC;

-- Remote employees, joined to their office location and monthly pay.
SELECT e.name,
       d.location,
       ROUND(e.salary / 12, 2) AS monthly
FROM employees e
JOIN departments d ON e.department = d.name
WHERE e.remote = true
ORDER BY monthly DESC NULLS LAST;

-- Distinct locations that have any remote staff.
SELECT DISTINCT d.location
FROM employees e
JOIN departments d ON e.department = d.name
WHERE e.remote = true
ORDER BY d.location;
