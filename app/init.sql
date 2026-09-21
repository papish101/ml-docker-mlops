CREATE TABLE IF NOT EXISTS predictions (

    id SERIAL PRIMARY KEY,

    sepal_length FLOAT,

    sepal_width FLOAT,

    petal_length FLOAT,

    petal_width FLOAT,

    prediction INTEGER,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
