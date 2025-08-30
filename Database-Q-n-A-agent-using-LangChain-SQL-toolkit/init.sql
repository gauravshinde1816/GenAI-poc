-- Create a sample table
CREATE TABLE IF NOT EXISTS employees (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    role VARCHAR(50) NOT NULL,
    salary DECIMAL(10,2)
);

-- Insert some sample data
INSERT INTO employees (name, role, salary) VALUES
('Alice', 'Engineer', 75000.00),
('Bob', 'Manager', 90000.00),
('Charlie', 'Intern', 30000.00);
