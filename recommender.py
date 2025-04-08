import pandas as pd
import random

from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import KNNWithMeans, SVD, NMF, accuracy

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, desc, lit, rand
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# Trainer class handles preprocessing, model training, and evaluation
class Trainer:  
    def __init__(self):
        # Initialize Spark session with reduced shuffle partitions
        self.spark = SparkSession.builder.config("spark.sql.shuffle.partitions", "200").getOrCreate()
        
    def get_preprocessed_data(self):
        print("--- Attempting To Load Spark Dataframe ---")
        try:
            # Try loading preprocessed data
            pd_events = pd.read_csv('Preprocessed Data/pd_events.csv')
            self.PandasData = pd_events
            self.create_surprise_dataset(pd_events)
            self.SparkData = self.spark.read.csv('Preprocessed Data/pd_events.csv', header=True, inferSchema=True)
            print("--- Spark Dataframe Loaded ---")
        except FileNotFoundError:
            # If not found, run preprocessing
            print("Events Dataframe not found. Preprocessing data...")
            self.preprocess_events()

    def preprocess_events(self):
        print("--- Start Spark Preprocessing ---")
        # Load data
        df_events = self.spark.read.csv('Data/events.csv', header=True, inferSchema=True)

        # Convert events to rating
        df_events = df_events.withColumn("rating", when(col("event") == "transaction", 5)
                                          .when(col("event") == "addtocart", 3)
                                          .when(col("event") == "view", 1)
                                          .otherwise(1))

        # Drop unnecessary columns
        df_events = df_events.drop(*["timestamp", "event", "transactionid"])
        self.SparkData = df_events
        print("--- Finish Spark Preprocessing ---")
        
        print("--- Converting Spark Dataframe to Pandas Dataframe ---")
        # Enable Arrow for faster conversion
        self.spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        pd_events = df_events.toPandas()
        pd_events.to_csv('Preprocessed Data/pd_events.csv', index=False)
        self.PandasData = pd_events
        self.create_surprise_dataset(pd_events)
        print("--- Finish Converting Spark Dataset to Pandas Dataframe ---")

    def create_surprise_dataset(self, pd_events):
        print("--- Creating Surprise Dataset ---")
        #Creating small sample dataset
        top_users = pd_events['visitorid'].value_counts().index[:30000]
        top_items = pd_events['itemid'].value_counts().index[:30000]
        sample_df = pd_events[pd_events['visitorid'].isin(top_users) & pd_events['itemid'].isin(top_items)]
        reader = Reader(rating_scale=(1, 5))
        sample_data = Dataset.load_from_df(sample_df, reader)
        self.SampleSurpriseData = sample_data

        # Create full dataset for SVD/NMF
        reader = Reader(rating_scale=(1, 5))
        full_data = Dataset.load_from_df(pd_events, reader)
        self.SurpriseData = full_data
        print("--- Finish Creating Surprise Dataset ---")

    def train_model(self, model_name):
        self.get_preprocessed_data()
        print(f"--- Start Training {model_name} Model---")
        # Select training method based on model_name
        match(model_name):
            case "als":
                model = self.ALS_trainer()
            case "knn-item":
                model = self.KNN_trainer(0)
            case "knn-user":
                model = self.KNN_trainer(1)
            case "svd":
                model = self.SVD_trainer()
            case "nmf":
                model = self.NMF_trainer()
        print(f"--- Finish Training {model_name} Model---")
        return model

    def ALS_trainer(self):
        # Initialize ALS model
        als = ALS(
                userCol="visitorid",
                itemCol="itemid",
                ratingCol="rating",
                implicitPrefs=False,
                coldStartStrategy="drop",  # Drop cold-start users
                rank=10,
                maxIter=5,
                regParam=0.1
            )
        sparkData = self.SparkData
        sparkData = sparkData.repartition(5)
        train_data, test_data = sparkData.randomSplit([0.8, 0.2], seed=42)
        model = als.fit(train_data)
        self.ALS_evaluator(model, test_data)
        return model
    
    def ALS_evaluator(self, model, test_data):
        # Evaluate ALS using RMSE and MAE
        predictions = model.transform(test_data)
        predictions = predictions.na.drop()

        rmse_evaluator = RegressionEvaluator(
            metricName="rmse", labelCol="rating", predictionCol="prediction"
        )
        rmse = rmse_evaluator.evaluate(predictions)

        mae_evaluator = RegressionEvaluator(
            metricName="mae", labelCol="rating", predictionCol="prediction"
        )
        mae = mae_evaluator.evaluate(predictions)

        print(f"RMSE ALS: {rmse}")
        print(f"MAE ALS: {mae}")

    def KNN_trainer(self, mode):
        # Train a KNN model (user/item-based based on mode)
        surprise_data = self.SampleSurpriseData
        trainset, testset = train_test_split(surprise_data, test_size=0.2)
        sim_options = {"name": "cosine", "user_based": bool(mode)}
        model = KNNWithMeans(k=10, sim_options=sim_options)
        model.fit(trainset)

        predictions = model.test(testset)
        accuracy.rmse(predictions)
        accuracy.mae(predictions)
        return model
    
    def SVD_trainer(self):
        # Train a SVD model
        surprise_data = self.SurpriseData
        trainset, testset = train_test_split(surprise_data, test_size=0.2)
        model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.1)
        model.fit(trainset)
        predictions = model.test(testset)

        accuracy.rmse(predictions)
        accuracy.mae(predictions)
        return model
    
    def NMF_trainer(self):
        # Train a NMF model
        surprise_data = self.SurpriseData
        trainset, testset = train_test_split(surprise_data, test_size=0.2)
        model = NMF(n_factors=20, n_epochs=50)
        model.fit(trainset)
        predictions = model.test(testset)
        
        accuracy.rmse(predictions)
        accuracy.mae(predictions)
        return model

    def get_spark_data(self):
        # Getter for Spark DataFrame
        return self.SparkData
    
    def get_pandas_data(self):
        # Getter for Pandas DataFrame
        return self.PandasData

# Recommender class handles generating and displaying recommendations
class Recommender:
    def __init__(self, model, model_name, spark_data, pandas_data):
        self.model = model
        self.model_name = model_name
        self.spark_data = spark_data
        self.pandas_data = pandas_data

    def recommend(self):
        while True:
            print("Select a user to recommend:"  )
            print("1. Random existing user")
            print("2. Fixed user 162285")
            print("3. A new user")
            print("4. Back to main menu")
            user = input("Enter choice: ")
            if user not in ["1", "2", "3", "4"]:
                print("Invalid input")
                continue
            if user == "4":
                break
            n = input("Enter number of recommendations (default 10): ")
            if not n:
                n = "10"
            if not n.isdigit():
                print("Invalid input")
                continue
            elif int(n) <= 0 or int(n) > 50:
                print("Number of recommendations must be between 1 and 50")
                continue
            if user == "1":
                self.recommend_existing_user(int(n))
            elif user == "2":
                self.recommend_existing_user(int(n), 162285)
            elif user == "3":
                self.recommend_new_user(int(n))
            
    def recommend_existing_user(self, n=10, default_user=None):
        # Generate recommendations for an existing user
        if self.model_name == "als":
            df_events = self.spark_data
            model = self.model
            if default_user:
                user_id = default_user
            else:
                user_id = df_events.select("visitorid").distinct().orderBy(rand()).limit(1).collect()[0][0]
            unique_item_ids = df_events.select("itemid").distinct()
            user_items_df = unique_item_ids.withColumn("visitorid", lit(user_id))

            # Generate recommendations using the model
            recommendations = model.transform(user_items_df)

            # Sort by prediction and select the top k
            top_k_recommendations = (
                recommendations.orderBy(desc("prediction")).limit(n).collect()
            )
            print(f"Top {n} Recommendations for User {user_id}:")
            for rec in top_k_recommendations:
                print(f"Item ID: {rec.itemid}, Prediction: {rec.prediction:0.2f}")
        else:
            if self.model_name == "knn-item" or self.model_name == "knn-user":
                df = self.pandas_data
                top_users = df['visitorid'].value_counts().index[:30000]
                top_items = df['itemid'].value_counts().index[:30000]
                df = df[df['visitorid'].isin(top_users) & df['itemid'].isin(top_items)]
            else:
                df = self.pandas_data
            model = self.model
            if default_user:
                user_id = default_user
            else:
                user_id = random.choice(df['visitorid'].unique())
            user_items = set(df[df['visitorid'] == user_id]['itemid'])
            all_items = set(df['itemid'].unique()) - user_items

            predictions = [(item, model.predict(user_id, item).est) for item in all_items]
            predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
            print(f"Top {n} Recommendations for User {user_id}:")
            for item, prediction in predictions:
                print(f"Item ID: {item}, Prediction: {prediction:0.2f}")

    def recommend_new_user(self, n=10):
        # Recommend items to a new user using popularity and randomness
        df_events = self.spark_data
        alpha = 0.7  # Adjust for more/less exploration (0 means fully random, 1 means fully popular)
        # Get popularity score equal to sum of ratings
        df = df_events.groupBy("itemid").agg({"rating": "sum"}).withColumnRenamed("sum(rating)", "popularity_score")
        # Normalize popularity score
        df = df.withColumn("normalized_popularity", col("popularity_score") / df.agg({"popularity_score": "max"}).collect()[0][0])
        # Add random factor which is a value between 0 and 1
        df = df.withColumn("random_factor", rand())
        # Calculate serendipitous score which is equal to alpha * normalized popularity score + (1 - alpha) * random factor
        df = df.withColumn("serendipitous_score", alpha * col("normalized_popularity") + (1 - alpha) * col("random_factor"))

        df = df.orderBy(col("serendipitous_score"), ascending=False)
        recommendations = df.limit(n)

        print(f"Top {n} Recommendations for New User:")
        for rec in recommendations.collect():
            print(f"Item ID: {rec.itemid}, Serendipitous Score: {rec.serendipitous_score:.2f}")
        
def get_model_input():
    while True:
        print("Choose a model to train:")
        print("1. ALS")
        print("2. KNN-item")
        print("3. KNN-user")
        print("4. SVD")
        print("5. NMF")
        model = input("Enter model: ")
        if model not in ["1", "2", "3", "4", "5"]:
            print("Invalid input")
            continue
        model_mapping = {"1": "als", "2": "knn-item", "3": "knn-user", "4": "svd", "5": "nmf"}
        model_name = model_mapping.get(model)
        return model_name

def main_cli():
    while True:
        print("Select an option")
        print("1. Train Model")
        print("2. Exit")
        choice = input("Choose an option: ")
        if choice not in ["1", "2"]:
            print("Invalid input")
            continue
        if choice == "1":
            model_name = get_model_input()
            trainer = Trainer()
            model = trainer.train_model(model_name)
            recommender = Recommender(model, model_name, trainer.get_spark_data(), trainer.get_pandas_data())
            recommender.recommend()
        elif choice == "2":
            print("Goodbye!")
            break

if __name__ == "__main__":
    main_cli()
