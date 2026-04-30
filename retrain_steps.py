from scipy.stats import pearsonr
from scipy.stats import ortho_group
from scipy.sparse import issparse
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectFromModel
from transformers import pipeline as transformer_pipeline
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os
import copy
import scipy.linalg as alg
import random
import argparse
import torch
import torch.nn as nn
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset


feature_labels = [
    "user_id",
    "public",
    "completion_percentage", 
    "gender", # 0,1
    "region", #categorical, string
    "last_login",
    "registration",
    "age", #numberical, 0-age attribute not set
    "body", #cm, kg, can also contain text
    "I_am_working_in_field", #text
    "spoken_languages", #text
    "hobbies", #text
    "I_most_enjoy_good_food", #text
    "pets", #text
    "body_type", #text
    "my_eyesight", #text
    "eye_color", #text
    "hair_color", #text
    "hair_type", #text
    "completed_level_of_education", #text
    "favourite_color", #text
    "relation_to_smoking", #text
    "relation_to_alcohol", #text
    "sign_in_zodiac", #text
    "on_pokec_i_am_looking_for", #text
    "love_is_for_me", #text
    "relation_to_casual_sex", #text
    "my_partner_should_be", #text
    "marital_status", #text
    "children", #text
    "relation_to_children", #text
    "I_like_movies", #text
    "I_like_watching_movie", #text
    "I_like_music", #text
    "I_mostly_like_listening_to_music", #text
    "the_idea_of_good_evening", #text
    "I_like_specialties_from_kitchen", #text
    "fun", #text, but contains a lot of link
    "I_am_going_to_concerts", #text
    "my_active_sports", #text
    "my_passive_sports", #text
    "profession", #text
    "I_like_books", #text
    "life_style", #text, jason file, contain links
    "music", #text, jason, link
    "cars", #text, jason, link
    "politics", #text, jason, link
    "relationships", #text, jason, link
    "art_culture", #text, jason, link
    "hobbies_interests", #text, jason, link
    "science_technologies", #text, jason, link
    "computers_internet", #text, jason, link
    "education", #text, jason, link
    "sport", #text, jason, link
    "movies", #text, jason, link
    "travelling", #text, jason, link
    "health", #text, jason, link
    "companies_brands", #text, jason, link
    "more" #text, jason, link
]

numerical_features = [
    "age"
    ]
categorical_features = [
    "gender",
    # "region"
    ]
textual_features = [
    # "body",
    # "I_am_working_in_field",
    # "spoken_languages",
    # "hobbies",
    # "I_most_enjoy_good_food",
    # "pets",
    # "body_type",
    # "my_eyesight",
    # "eye_color",
    # "hair_color",
    # "hair_type",
    # "completed_level_of_education",
    # "favourite_color",
    # "relation_to_smoking",
    "relation_to_alcohol",
    # "sign_in_zodiac",
    # "on_pokec_i_am_looking_for",
    # "love_is_for_me",
    # "relation_to_casual_sex",
    # "my_partner_should_be",
    # "marital_status",
    # "children",
    # "relation_to_children",
    # "I_like_movies",
    # "I_like_watching_movie",
    # "I_like_music",
    # "I_mostly_like_listening_to_music",
    # "the_idea_of_good_evening",
    # "I_like_specialties_from_kitchen",
    # "fun",
    # "I_am_going_to_concerts",
    # "my_active_sports",
    # "my_passive_sports",
    # "profession",
    # "I_like_books"
    # "life_style",
    # "music",
    # "cars",
    # "politics",
    # "relationships",
    # "art_culture",
    # "hobbies_interests",
    # "science_technologies",
    # "computers_internet",
    # "education",
    # "sport",
    # "movies",
    # "travelling",
    # "health",
    # "companies_brands",
    # "more"
]




class TextConcatEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name, batch_size=16, device=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model = None

    def fit(self, X, y=None):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            rows = X.astype(str).fillna("").agg(" ".join, axis=1).tolist()
        else:
            rows = pd.DataFrame(X).astype(str).fillna("").agg(" ".join, axis=1).tolist()
        rows = [s.strip() for s in rows]
        return np.asarray(
            self.model.encode(
                rows,
                batch_size=self.batch_size,
                show_progress_bar=True,
            )
        )


def build_pipeline(numerical_features, categorical_features, filtered_textual_features, model_name, batch_size, device):
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    text_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="")),
            ("embed", TextConcatEmbedder(model_name, batch_size=batch_size, device=device)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, numerical_features),
            ("cat", cat_transformer, categorical_features),
            ("text", text_transformer, filtered_textual_features),
        ],
        sparse_threshold=0.3,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="kinit/slovakbert-sts-stsb")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu")
    return parser.parse_args()


def preprocess(df, target_column, edge_path="pokec_dataset/relationships.txt"):

    # we only focus individuals with public friendships
    df_public = df[df["public"] != 0]
    df_public_ids = set(df_public["user_id"].values)
    edges = pd.read_csv(edge_path, sep="\t", header=None)
    network_g = nx.Graph()
    # print("number of users with public friendships:", len(df_public_ids))
    for edge in edges.itertuples():
        if edge[1] not in df_public_ids or edge[2] not in df_public_ids:
            continue 
        else:
            network_g.add_edge(edge[1], edge[2])
    
    components = list(nx.connected_components(network_g))
    lcc = max(components, key=len)
    network_lcc = network_g.subgraph(lcc).copy()
    with open("pokec_dataset/lcc_graph_" + target_column + ".pk", "wb") as f:
        pickle.dump(network_lcc, f)
    lcc_public_df = df_public[df_public["user_id"].isin(lcc)]
    print("number of users in lcc with public friendships:", len(lcc_public_df))
    nodelist = list(lcc_public_df["user_id"].values)  # 0..n-1 in row order
    w = nx.to_numpy_array(network_lcc, nodelist=nodelist, dtype=int)
    return lcc_public_df, network_lcc

def sentiment_scores(texts, sentiment_pipe, batch_size=32):
    scores = sentiment_pipe(texts, batch_size=batch_size)
    return np.array([
        r["score"] if r["label"] == "positive"
        else 0.5 if r["label"] == "neutral"
        else 1 - r["score"]
        for r in scores
    ], dtype=float)


def extract_features(df, args, filtered_textual_features, numerical_features_extended):
    
    
    # enforce data types
    df[numerical_features_extended] = df[numerical_features_extended].apply(pd.to_numeric, errors="coerce")
    df[categorical_features] = df[categorical_features].astype(str)
    df[filtered_textual_features] = df[filtered_textual_features].astype(str)

    use_sentiment_scores = True
    if use_sentiment_scores:

        # sentiment model (same as y_label)
        sentiment = transformer_pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            device=0 if args.device and "cuda" in args.device else -1,
            use_fast=False,
        )

        text_feat_matrix = {}
        for col in filtered_textual_features:
            text_feat_matrix[f"{col}_sent"] = sentiment_scores(
                df[col].fillna("").astype(str).tolist(),
                sentiment
            )

        text_feat_df = pd.DataFrame(text_feat_matrix, index=df.index)

        # numeric + categorical
        num_df = df[numerical_features].apply(pd.to_numeric, errors="coerce").fillna(0)
        cat_df = df[categorical_features].astype(str)

        # one-hot categorical
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_ohe = ohe.fit_transform(cat_df)
        cat_ohe_df = pd.DataFrame(cat_ohe, index=df.index, columns=ohe.get_feature_names_out())

        # final feature matrix
        X_features = pd.concat([num_df, cat_ohe_df, text_feat_df], axis=1)
        X_features = X_features.to_numpy()
        print("Feature matrix shape:", X_features.shape)

    else:

        pipeline = build_pipeline(
            numerical_features_extended,
            categorical_features,
            filtered_textual_features,
            model_name=args.model,
            batch_size=args.batch_size,
            device=args.device,
        )

        
        X = df[numerical_features_extended + categorical_features + filtered_textual_features]

        X_features = pipeline.fit_transform(X)  


    print("Feature matrix shape:", X_features.shape)
    return X_features


def compute_score(df, target_column, args):
    
    sentiment = transformer_pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        device=0 if args.device and "cuda" in args.device else -1,
    )
    
    scores = sentiment(df[target_column].fillna("").astype(str).tolist(), batch_size=32)
    results = [
        r["score"] if r["label"] == "positive"
        else 0.5 if r["label"] == "neutral"
        else 1 - r["score"]
        for r in scores
    ]
    
    return results

def add_graph_features(df, graph_path):
    
    with open(graph_path, "rb") as f:
        network_lcc = pickle.load(f)
    
    degree = dict(network_lcc.degree())
    clustering = nx.clustering(network_lcc)
    pagerank = nx.pagerank(network_lcc, alpha=0.85)

    df["deg"] = df["user_id"].map(degree).fillna(0)
    df["clust"] = df["user_id"].map(clustering).fillna(0)
    df["pr"] = df["user_id"].map(pagerank).fillna(0)
    return df

class SigmoidMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def predicting(model_name, X_features_labeled, y_label, X_features_unlabeled):
    

    X_train, X_test, y_train, y_test = train_test_split(
        X_features_labeled, y_label, test_size=0.2, random_state=2026
    )
    if model_name == "neural_net":
        if issparse(X_features_labeled):
            X_features_labeled = X_features_labeled.toarray()
        if issparse(X_features_unlabeled):
            X_features_unlabeled = X_features_unlabeled.toarray()
        # standardize

        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True) + 1e-6
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                torch.tensor(y_train, dtype=torch.float32))
        test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                torch.tensor(y_test, dtype=torch.float32))

        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        model = SigmoidMLP(X_train.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        # train
        for epoch in range(20):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()

        # eval
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
            preds = np.array(preds)
            unlabeled_preds = model(torch.tensor(X_features_unlabeled, dtype=torch.float32)).numpy()
            unlabeled_preds = np.array(unlabeled_preds)
        rmse = np.sqrt(np.mean((preds - y_test) ** 2))
        r2 = 1 - np.sum((preds - y_test) ** 2) / np.sum((y_test - np.array(y_test).mean()) ** 2)

        print(f"neural net, RMSE: {rmse:.4f} | R2: {r2:.4f}")

    elif model_name == 'ridge':
        
        model = Ridge(alpha=0.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0.0, 1.0)
        unlabeled_preds = model.predict(X_features_unlabeled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"Odinary least square with clipping regression, RMSE: {rmse:.4f} | R2: {r2:.4f}")
    else:
        # mean estimation
        y_pred = np.mean(y_train) * np.ones_like(y_test)
        mean_rmse = np.sqrt(np.mean((y_test - np.mean(y_train)) ** 2))
        print(f"predicting mean RMSE: {mean_rmse:.4f}")
        unlabeled_preds = np.mean(y_label) * np.ones(len(X_features_unlabeled))

    return unlabeled_preds

def run_simulation(
        network,
        nodelist,
        platform_params,
        peer_params,
        steer_nodes,
        retrain_steps,
        fj_steps,
        x_star,
        policy,
        model_name,
        X_features_labeled,
        X_features_unlabeled
):
    agent_num = len(x_star)
    n = int(agent_num * 0.8)
    adj_mat = nx.to_numpy_array(network, nodelist=nodelist)
    weight_mat = copy.deepcopy(adj_mat)
    platform_params_gamma0 = copy.deepcopy(platform_params)
    for i in range(agent_num):
        if i not in steer_nodes:
            platform_params_gamma0[i] = 0.

    degs_inv = 1/np.sum(adj_mat, axis=0)
    for i in range(agent_num):
        if np.isinf(degs_inv[i]):
            degs_inv[i] = 0.
        elif degs_inv[i] > 1.1:
            degs_inv[i] = 0.
    
    whole_record = np.zeros((agent_num, retrain_steps+1))
    x_labeled_prior = x_star[:n].copy()
    x_unlabeled_prior = x_star[n:].copy()
    x_labeled_prior_gamma0 = x_star[:n].copy()
    x_unlabeled_prior_gamma0 = x_star[n:].copy()
    whole_record[:, 0] = copy.deepcopy(x_star)
    whole_record_gamma0 = np.zeros((agent_num, retrain_steps+1))
    whole_record_gamma0[:, 0] = copy.deepcopy(x_star)
    platform_predictions = np.zeros(agent_num)
    platform_predictions_gamma0 = np.zeros(agent_num)
    for t in range(retrain_steps):
         
        if policy == 'sl':
            # supervised learning policy
            # assume the initial predictions are the innate opinions
            platform_predictions[:n] = copy.deepcopy(x_labeled_prior)
            platform_predictions_gamma0[:n] = copy.deepcopy(x_labeled_prior_gamma0)
            if model_name == 'perfect':
                platform_predictions[n:] = copy.deepcopy(x_unlabeled_prior)
                platform_predictions_gamma0[n:] = copy.deepcopy(x_unlabeled_prior_gamma0)
            else:
                platform_predictions[n:] = predicting(model_name, X_features_labeled, x_labeled_prior, X_features_unlabeled)
                platform_predictions_gamma0[n:] = predicting(model_name, X_features_labeled, x_labeled_prior_gamma0, X_features_unlabeled)
            
        else:
            # steering policy

            platform_predictions[:n] = copy.deepcopy(x_labeled_prior)
            platform_predictions_gamma0[:n] = copy.deepcopy(x_labeled_prior_gamma0)
            if model_name == 'perfect':
                platform_predictions[n:] = copy.deepcopy(x_unlabeled_prior)
                platform_predictions_gamma0[n:] = copy.deepcopy(x_unlabeled_prior_gamma0)
            else: 
                platform_predictions[n:] = predicting(model_name, X_features_labeled, x_labeled_prior, X_features_unlabeled)
                platform_predictions_gamma0[n:] = predicting(model_name, X_features_labeled, x_labeled_prior_gamma0, X_features_unlabeled)
            # we only steer on one node towards 1.
            for node in steer_nodes:
                platform_predictions[node] = 1. 
                platform_predictions_gamma0[node] = 1.

        x_zero = np.diag(np.ones(agent_num) - platform_params) @ x_star + platform_params * platform_predictions 
        x_zero_gamma0= np.diag(np.ones(agent_num) - platform_params_gamma0) @ x_star + platform_params_gamma0 * platform_predictions
        x_temp = copy.deepcopy(x_zero)
        x_temp_gamma0= copy.deepcopy(x_zero_gamma0)
        
        for k in range(fj_steps):

            x_temp = (np.ones(agent_num) - peer_params) * x_zero + np.diag(peer_params) @ np.diag(degs_inv) @ weight_mat @ x_temp
            x_temp_gamma0 = (np.ones(agent_num) - peer_params) * x_zero_gamma0 + np.diag(peer_params) @ np.diag(degs_inv) @ weight_mat @ x_temp_gamma0
            
        
        whole_record[:, t+1] = copy.deepcopy(x_temp)
        whole_record_gamma0[:, t+1] = copy.deepcopy(x_temp_gamma0)
        x_labeled_prior = copy.deepcopy(x_temp[:n])
        x_labeled_prior_gamma0 = copy.deepcopy(x_temp_gamma0[:n])
        x_unlabeled_prior = copy.deepcopy(x_temp[n:])
        x_unlabeled_prior_gamma0 = copy.deepcopy(x_temp_gamma0[n:])
    return whole_record, whole_record_gamma0




def run_opinion_dynamics(innate_opinions, network_lcc, nodelist, model_name, X_features_labeled, X_features_unlabeled, policy, strong_perform):
    
    agent_num = len(innate_opinions)
    # this is to reveal the steer effect on the stubborn node.
    # innate_opinions = np.zeros(agent_num)
    fj_K = 100
    retrain_T = 4
    x_initial = copy.deepcopy(innate_opinions)

    param_folder = "pokec_dataset/parametric_params/"
    realworld_params = Path(param_folder)
    realworld_params.mkdir(exist_ok=True)

    # generate heterogeneous parameters
    if policy == "steer":
        platform_file_path = Path(param_folder + "single_steer_platform_sus" + str(agent_num) + ".pkl")
    else:
        platform_file_path = Path(param_folder + "hetero_platform_sus" + str(agent_num) + ".pkl")

    if not platform_file_path.exists():

        if policy == "steer":
            # use the same platform susceptibilities except for the stubborn node.
            ori_platform_file_path = Path(param_folder + "hetero_platform_sus" + str(agent_num) + ".pkl")
            with open(ori_platform_file_path, "rb") as file:
                platform_sus = pickle.load(file)
            steer_size = int(agent_num / 10)
            selected_nodes = np.random.choice(agent_num, size=steer_size+1, replace=False)
            steer_nodes = selected_nodes[:-1]
            stubborn_node = selected_nodes[-1]
            platform_sus[stubborn_node] = 0.
            with open(platform_file_path, "wb") as file:
                pickle.dump(platform_sus, file)
            with open(param_folder + "steer_node_" + str(agent_num) + ".pkl", "wb") as file:
                pickle.dump(steer_nodes, file)
            with open(param_folder + "stubborn_node_" + str(agent_num) + ".pkl", "wb") as file:
                pickle.dump(stubborn_node, file)
        else: 
            platform_sus = np.clip(np.random.normal(loc=0.9, scale=0.1, size=agent_num), 0.01, 0.99)
            with open(platform_file_path, "wb") as file:
                pickle.dump(platform_sus, file)

            steer_nodes = []
            stubborn_node = None
    else:
        with open(platform_file_path, "rb") as file:
            platform_sus = pickle.load(file)
        if policy == "steer":
            with open(param_folder + "steer_node_" + str(agent_num) + ".pkl", "rb") as file:
                steer_nodes = pickle.load(file)
            with open(param_folder + "stubborn_node_" + str(agent_num) + ".pkl", "rb") as file:
                stubborn_node = pickle.load(file)
        else:
            steer_nodes = []
            stubborn_node = None

    peer_file_path = Path(param_folder + "hetero_peer_sus" + str(agent_num) + ".pkl")

    if not peer_file_path.exists():
        
        peer_sus = np.clip(np.random.normal(loc=0.9, scale=0.1, size=agent_num), 0.01, 0.99)
        with open(peer_file_path, "wb") as file:
            pickle.dump(peer_sus, file)
    else:
        with open(peer_file_path, "rb") as file:
            peer_sus = pickle.load(file)

    
    if strong_perform:
        results_folder = "pokec_dataset/results_strong_perform/"
        platform_sus = np.ones(agent_num)
    else:
        results_folder = "pokec_dataset/results/"
    if os.path.exists(results_folder + model_name + "_" + policy + "_whole_record" + str(retrain_T) + ".pk"):
        with open(results_folder + model_name + "_" + policy + "_whole_record" + str(retrain_T) + ".pk", "rb") as f:
            whole_opinions = pickle.load(f)
       
    else:

        whole_opinions, whole_opinions_gamma0 = run_simulation(network=network_lcc, nodelist=nodelist, platform_params=platform_sus, 
                                            peer_params=peer_sus, 
                                            steer_nodes=steer_nodes, fj_steps=fj_K, retrain_steps=retrain_T, 
                                            x_star=innate_opinions, policy=policy, model_name=model_name, 
                                            X_features_labeled=X_features_labeled, X_features_unlabeled=X_features_unlabeled)
        
        with open(results_folder + model_name + "_" + policy + "_whole_record" + str(retrain_T) + ".pk", "wb") as f:
            pickle.dump(whole_opinions, f)
        with open(results_folder + model_name + "_" + policy + "_gamma0_whole_record" + str(retrain_T) + ".pk", "wb") as f:
            pickle.dump(whole_opinions_gamma0, f)

    # FJ(x^*) - needs to be pre-generated - no steering
    with open("pokec_dataset/results/" + model_name + "_FJequilibrium.pk", "rb") as f:
        FJ_equilibrium = pickle.load(f)

    if policy == "steer":
        
        x = np.arange(1, retrain_T+1)
        plt.plot(x, whole_opinions[stubborn_node, 1:], label=r"$(x_{PS})_l$")
        print(FJ_equilibrium[stubborn_node])
        plt.hlines(y=FJ_equilibrium[stubborn_node], xmin=1, xmax=retrain_T, linestyle='--', label=r"(FJ($x^*$)$)_l$")
        plt.hlines(y=whole_opinions_gamma0[stubborn_node, -1], xmin=1, xmax=retrain_T, linestyle='--', label=r"$(x_{ex}^{(T)})_l$ ($\gamma_k=0,k\neq j,l$)", color='orange')
        plt.xticks(range(1, retrain_T+1))
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        plt.ylabel("Opinion", fontsize=13)
        plt.legend(loc="upper left", bbox_to_anchor=(1,1), frameon=False, fontsize=10)
        plt.savefig(param_folder + model_name + "_parametric_steer_retrain_steps.pdf", bbox_inches='tight')

    


def plot_adjust(innate_opinions, policy, strong_perform):
    agent_num = len(innate_opinions)
    retrain_T = 4
    if strong_perform:
        results_folder = "pokec_dataset/results_strong_perform/"
    else:
        results_folder = "pokec_dataset/results/"
    param_folder = "pokec_dataset/parametric_params/"

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    models = ["perfect", "ridge", "neural_net", "mean"]

   
    x = np.arange(0, retrain_T+1)
    if policy == "steer":
    
        labels = ["Perfect", "OLS", "MLP", "Mean"]

        with open(param_folder + "stubborn_node_" + str(agent_num) + ".pkl", "rb") as file:
            stubborn_node = pickle.load(file)

        with open("pokec_dataset/results/" + "perfect_FJequilibrium.pk", "rb") as f:
            FJ_equilibrium = pickle.load(f)

        with open(results_folder + "perfect_steer_gamma0_whole_record" + str(retrain_T) + ".pk", "rb") as f:
            x_psl_gamma0 = pickle.load(f)
            x_psl_gamma0 = x_psl_gamma0[stubborn_node, -1]

        plt.hlines(y=x_psl_gamma0, xmin=0, xmax=retrain_T, linestyle='--', label=r"$(x_{ex}^{(T)})_l$" + "(Perfect,\n" + r"$\beta_k=0,k\notin \{l\}\cup S$)", color='purple')
        
        for i in range(len(models)):
            
            if os.path.exists(results_folder + models[i] + "_steer_whole_record" + str(retrain_T) + ".pk"):
                with open(results_folder + models[i] + "_steer_whole_record" + str(retrain_T) + ".pk", "rb") as f:
                    whole_opinions = pickle.load(f)
            
            plt.plot(x, whole_opinions[stubborn_node, :], label=labels[i], color=colors[i])
            print(FJ_equilibrium[stubborn_node])

        plt.xticks(range(0, retrain_T+1, 5), fontsize=12)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        plt.ylabel(r"Opinion $(x_{ex}^{(t)})_l$", fontsize=18)
        plt.xlabel(r"Retraining step $t$", fontsize=18)
        plt.yticks(fontsize=12)
        plt.legend(loc="upper left", bbox_to_anchor=(0,1), frameon=False, fontsize=12)
        
        plt.savefig(param_folder + "all_parametric_steer_retrain_steps.pdf", bbox_inches='tight')
    else:
        # for supervised learning policy 
        mean_labels = [r"Mean($x_{ex}^{(t)}$) (Perfect prediction)", r"Mean($x_{ex}^{(t)}$) (OLS)", r"Mean($x_{ex}^{(t)}$) (MLP)", r"Mean($x_{ex}^{(t)}$) (Mean estimation)"]
        std_labels = [r"Var($x_{ex}^{(t)}$) (Perfect prediction)", r"Var($x_{ex}^{(t)}$) (OLS)", r"Var($x_{ex}^{(t)}$) (MLP)", r"Var($x_{ex}^{(t)}$) (Mean estimation)"]
        labels = ["Perfect", "OLS", "MLP", "Mean"]
        
        fig, ax = plt.subplots()
        step_gap = 15
        box_group_width = 7.5
        
        positions_base = np.arange(retrain_T + 1) * step_gap
        offsets = np.linspace(
            -box_group_width / 2, box_group_width / 2, len(models), endpoint=False
        ) + (box_group_width / len(models)) / 2
        box_width = 0.85 * (box_group_width / len(models))

        df = {}
        for i in range(len(models)):
            
            
            if os.path.exists(results_folder + models[i] + "_" + policy + "_whole_record" + str(retrain_T) + ".pk"):
                with open(results_folder + models[i] + "_" + policy + "_whole_record" + str(retrain_T) + ".pk", "rb") as f:
                    whole_opinions = pickle.load(f)
                
                df[labels[i]] = whole_opinions[:, :]
                
                
        
        expanded_rows = []
        for model_name, temp_opinions in df.items():
            temp_df = pd.DataFrame(temp_opinions.T)
            temp_df_expanded = temp_df.melt(var_name="sample", value_name="value", ignore_index=False)
            temp_df_expanded = temp_df_expanded.rename_axis("time").reset_index()
            temp_df_expanded["model"] = model_name
            expanded_rows.append(temp_df_expanded)
        
        all_rows = pd.concat(expanded_rows, ignore_index=True)

        stats = (
            all_rows.groupby(["time", "model"])["value"]
            .agg(mean="mean", var="var")
            .reset_index()
        )
        stats["std"] = np.sqrt(stats["var"])   # use variance-derived error bars


        models_u = [m for m in labels if m in stats["model"].unique()]
        print(models_u)


        for m in models_u:
            s = stats[stats["model"] == m].copy()
            i = labels.index(m)
            x = positions_base + offsets[i]
            ax.errorbar(
                x, s["mean"], yerr=s["std"],
                fmt="s",            # square marker ("box")
                linestyle="none",   # no line between steps
                elinewidth=box_width*0.4,       # error bar line width
                capthick=box_width*0.25,      # error bar cap thickness
                markeredgewidth=box_width*0.4,  # marker edge width
                markersize=box_width*0.5,       # marker size
                capsize=box_width*0.4,
                label=m,
                color=colors[i]
            )
        
        ax.set_xticks(positions_base[::5])
        ax.set_xticklabels(np.arange(retrain_T + 1)[::5], fontsize=12)

        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        plt.ylabel(r"Opinion after peer interaction, $x_{ex}^{(t)}$", fontsize=15)
        plt.xlabel(r"Retraining step $t$", fontsize=18)
        plt.yticks(fontsize=12)
        
        plt.legend(loc="upper right", bbox_to_anchor=(1,1), frameon=False, fontsize=15, columnspacing=0.2, labelspacing=0.2, borderpad=0.2, handletextpad=0.2)
             
        plt.savefig(param_folder + "all_parametric_sl_retrain_steps.pdf", bbox_inches='tight')

def main():
    args = parse_args()

    target_column = "relation_to_smoking"

    if os.path.exists("pokec_dataset/lcc_profiles_" + target_column + ".pk"):
        with open("pokec_dataset/lcc_profiles_" + target_column + ".pk", "rb") as f:
            df = pickle.load(f)
        print("lcc user num:", len(df))

        with open("pokec_dataset/lcc_graph_" + target_column + ".pk", "rb") as f:
            network_lcc = pickle.load(f)
    else:
        df = pd.read_csv("pokec_dataset/profiles.txt", sep="\t", header=None)
        feature_len = len(feature_labels)
        df = df.iloc[:, :feature_len]
        df.columns = feature_labels
        df = df.replace(r"^\s*$", np.nan, regex=True)
        mask = df[target_column].isna() | (df[target_column].astype(str).str.strip() == "")
        df = df[~mask].copy()
        df = df.sample(n=30000, random_state=2026)
        # now only users in lcc and public friendships are maintained
        df, network_lcc = preprocess(df, target_column, edge_path="pokec_dataset/relationships.txt")
        print("number of nodes in lcc:", len(network_lcc))
        with open("pokec_dataset/lcc_profiles_" + target_column + ".pk", "wb") as f:
            pickle.dump(df, f)
    
    include_graph_features = False
    if include_graph_features:
        df = add_graph_features(df, graph_path="pokec_dataset/lcc_graph_" + target_column + ".pk")
        numerical_features_extended = numerical_features + ["deg", "clust", "pr"]
    else: 
        numerical_features_extended = numerical_features.copy()
    
    # assume we don't have access to the label of the 20% population
    n = int(len(df) * 0.8) 
    df_labeled = df.iloc[:n].copy()
    df_unlabeled = df.iloc[n:].copy()
    # we compute the sentiment scores as innate opinions of individuals
    # the platform has access to the innate opinion of labeled group
    # the platform has no access to the innate opinion of labeled group
    if not os.path.exists("pokec_dataset/parametric_params/y_label" + str(len(df)) + ".pk"):
        y_label = compute_score(df_labeled, target_column, args)
        y_unlabel_label = compute_score(df_unlabeled, target_column, args)
        with open("pokec_dataset/parametric_params/y_label" + str(len(df)) + ".pk", "wb") as f:
            pickle.dump(y_label, f)
        with open("pokec_dataset/parametric_params/y_unlabel_label" + str(len(df)) + ".pk", "wb") as f:
            pickle.dump(y_unlabel_label, f)
    else:
        with open("pokec_dataset/parametric_params/y_label" + str(len(df)) + ".pk", "rb") as f:
            y_label = pickle.load(f)
        with open("pokec_dataset/parametric_params/y_unlabel_label" + str(len(df)) + ".pk", "rb") as f:
            y_unlabel_label = pickle.load(f)
   
    # extract features from mixed data types
    filtered_textual_features = [text for text in textual_features if text != target_column]
    df_labeled[numerical_features_extended] = df_labeled[numerical_features_extended].apply(pd.to_numeric, errors="coerce")
    df_labeled[categorical_features] = df_labeled[categorical_features].astype(str)
    df_labeled[filtered_textual_features] = df_labeled[filtered_textual_features].astype(str)
    df_unlabeled[numerical_features_extended] = df_unlabeled[numerical_features_extended].apply(pd.to_numeric, errors="coerce")
    df_unlabeled[categorical_features] = df_unlabeled[categorical_features].astype(str)
    df_unlabeled[filtered_textual_features] = df_unlabeled[filtered_textual_features].astype(str)

    
    
    if os.path.exists("pokec_dataset/labeled_feature_matrix_" + target_column + "_" + str(include_graph_features) + ".pk"):
        print("Loading features")
        with open("pokec_dataset/labeled_feature_matrix_" + target_column + "_" + str(include_graph_features) + ".pk", "rb") as f:
            X_features_labeled = pickle.load(f)
        print("Feature matrix shape:", X_features_labeled.shape)
        with open("pokec_dataset/unlabeled_feature_matrix_" + target_column + "_" + str(include_graph_features) + ".pk", "rb") as f:
            X_features_unlabeled = pickle.load(f)
    else:
        print("Extracting features")
        X_features_labeled = extract_features(df_labeled, args, filtered_textual_features, numerical_features_extended)
        X_features_unlabeled = extract_features(df_unlabeled, args, filtered_textual_features, numerical_features_extended)
        with open("pokec_dataset/labeled_feature_matrix_" + target_column + "_" + str(include_graph_features) + ".pk", "wb") as f:
            pickle.dump(X_features_labeled, f)
        with open("pokec_dataset/unlabeled_feature_matrix_" + target_column + "_" + str(include_graph_features) + ".pk", "wb") as f:
            pickle.dump(X_features_unlabeled, f)
    
    model_name = "mean"  # "neural_net" or "ridge" or "mean" or "perfect"
    
    # computed sentiment scores are assumed to be innate opinions, x_star
    innate_opinions = np.array(y_label + y_unlabel_label)
    adjust_plot = True
    policy = "sl"  # "sl" for supervised learning, "steer" for steering
    strong_perform = True  # when it's true, platform_sus = 1 for all individuals
    if adjust_plot:
        plot_adjust(innate_opinions, policy, strong_perform)
    else: 
        for model_name in ["perfect", "ridge", "neural_net", "mean"]:
            run_opinion_dynamics(innate_opinions, network_lcc, df["user_id"].values, model_name, X_features_labeled, X_features_unlabeled, policy, strong_perform)




if __name__ == "__main__":
    main()
