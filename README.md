# Personality Prediction Project with MLOps Implementation

This project combines Data Science with MLOps principles to create an automated, scalable framework for personality prediction. By leveraging modern operations practices, we ensure that the model development and deployment processes are efficient, robust, and repeatable.

## Overview

The Personality Prediction Project employs machine learning algorithms to classify personality types (Introvert vs. Extrovert). This initiative extends beyond traditional modeling; it implements a comprehensive end-to-end MLOps pipeline. The project highlights the importance of automating the ML lifecycle, focusing on streamlining development, facilitating smooth deployment, and ensuring long-term maintainability.


## 📁 Project Setup and Structure
#### Step 1: Project Template
    * Start by executing the template.py file to create the initial project template, which includes the required folder structure and placeholder files.
#### Step 2: Package Management
    * Write the setup for importing local packages in setup.py and pyproject.toml files.

#### Step 3: Virtual Environment and Dependencies
    * Create a virtual environment and install required dependencies from requirements.txt
      python -m venv (name of venv)
    * Using the Activate.ps1 activate the virtual environment
       pip install - requirements.txt

    * verfiy the local packages by running:
       pip list

## 📊 MongoDB Setup and Data Management

#### Step 4: MongoDB Atlas Configuration
    1. Sign up for MongoDB Atlas and create a new project.
    2. Set up a free M0 cluster, configure the username and password, and allow access from any IP address (0.0.0.0/0).
    3. Retrieve the MongoDB connection string for Python and save it (replace <password> with your password).
#### Step 5: Pushing Data to MongoDB
    1. Create a folder named notebook, add the dataset, and create a notebook file mongoDB_demo.ipynb.
    2. Use the notebook to push data to the MongoDB database.
    3. Verify the data in MongoDB Atlas under Database > Data Explore.