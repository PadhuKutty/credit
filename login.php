<?php
// Database connection
$servername = "localhost";
$username = "Padma";
$password = "Priya@2004";
$dbname = "Creditcard";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Check if form is submitted
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // Get form data
    $email = $_POST["email"];
    $password = $_POST["password"];

    // Check if email and password match
    $sql = "SELECT * FROM users WHERE email='$email' AND password='$password'";
    $result = $conn->query($sql);

    if ($result->num_rows > 0) {
        // Login successful
        echo "Login successful";
    } else {
        // Login failed
        echo "Invalid email or password";
    }
}

// Close connection
$conn->close();
?>
