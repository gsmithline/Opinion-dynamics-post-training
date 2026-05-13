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

from split_helper import get_new_order, get_n_labeled, split_suffix
from subgroup_helper import Subgroup
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


def predicting(model_name, X_features_labeled, y_label, X_features_unlabeled,
               df_labeled=None, df_unlabeled=None, sim_params=None):

    if model_name == "llm":
        from llm_predictor import predicting_llm
        return predicting_llm(df_labeled, y_label, df_unlabeled, sim_params=sim_params)

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
        steering_params,
        steering_vector,
        retrain_steps,
        fj_steps,
        x_star,
        policy,
        model_name,
        X_features_labeled,
        X_features_unlabeled,
        df_labeled=None,
        df_unlabeled=None,
):
    agent_num = len(x_star)
    n = int(agent_num * 0.8)
    adj_mat = nx.to_numpy_array(network, nodelist=nodelist)
    weight_mat = copy.deepcopy(adj_mat)

    degs_inv = 1/np.sum(adj_mat, axis=0)
    for i in range(agent_num):
        if np.isinf(degs_inv[i]):
            degs_inv[i] = 0.
        elif degs_inv[i] > 1.1:
            degs_inv[i] = 0.
    
    whole_record = np.zeros((agent_num, retrain_steps+1))
    x_labeled_prior = x_star[:n].copy()
    x_unlabeled_prior = x_star[n:].copy()
    whole_record[:, 0] = copy.deepcopy(x_star)
    platform_predictions = np.zeros(agent_num)

    x_star_mean = float(np.mean(x_star))
    x_star_std = float(np.std(x_star))
    sim_params = {
        "agent_num": agent_num,
        "n_labeled": n,
        "n_unlabeled": agent_num - n,
        "retrain_steps": retrain_steps,
        "fj_steps": fj_steps,
        "policy": policy,
        "model_name": model_name,
        "platform_sus_mean": float(np.mean(platform_params)),
        "platform_sus_std": float(np.std(platform_params)),
        "peer_sus_mean": float(np.mean(peer_params)),
        "peer_sus_std": float(np.std(peer_params)),
        "steer_strength_mean": float(np.mean(steering_params)),
        "steer_strength_std": float(np.std(steering_params)),
        "x_star_mean": x_star_mean,
        "x_star_std": x_star_std,
    }

    # Subgroup tracking. TRACK_SUBGROUP takes precedence; otherwise falls back
    # to SFT_SUBGROUP so a sliver run automatically tracks its own subgroup.
    # When neither is set, _tracker is None and per-round logging adds no keys.
    _tracker: Subgroup | None = (
        Subgroup.from_env("TRACK_SUBGROUP") or Subgroup.from_env("SFT_SUBGROUP")
    )
    _track_mask_full: np.ndarray | None = None
    _track_mask_labeled: np.ndarray | None = None
    _track_mask_unlabeled: np.ndarray | None = None
    if _tracker is not None and df_labeled is not None and df_unlabeled is not None:
        _df_full = pd.concat(
            [df_labeled.reset_index(drop=True),
             df_unlabeled.reset_index(drop=True)],
            ignore_index=True,
        )
        _track_mask_full = _tracker.compute_mask(_df_full)
        _track_mask_labeled = _track_mask_full[:n]
        _track_mask_unlabeled = _track_mask_full[n:]
        sim_params["track_subgroup_tag"] = _tracker.tag
        sim_params["track_n_S"] = int(_track_mask_full.sum())
        sim_params["track_n_S_labeled"] = int(_track_mask_labeled.sum())
        sim_params["track_n_S_unlabeled"] = int(_track_mask_unlabeled.sum())
        print(f"[track] subgroup={_tracker.tag}  |S|={int(_track_mask_full.sum())} "
              f"(labeled={int(_track_mask_labeled.sum())}, "
              f"unlabeled={int(_track_mask_unlabeled.sum())})")

    for t in range(retrain_steps):
        print(f"[run_simulation] round {t+1}/{retrain_steps} starting ({model_name})", flush=True)

        # Log input-target distribution stats BEFORE SFT (decouples FJ-compression from LLM-compression)
        try:
            import wandb
            if wandb.run is not None:
                target_payload = {
                    "round": t + 1,
                    "target_std": float(np.std(x_labeled_prior)),
                    "target_mean": float(np.mean(x_labeled_prior)),
                    "target_range": float(np.max(x_labeled_prior) - np.min(x_labeled_prior)),
                }
                # Per-subgroup target_mean: pre-SFT, labeled-side analog of the existing
                # target_mean. Adds nothing when no tracker is set.
                if _track_mask_labeled is not None:
                    sl = _track_mask_labeled
                    if sl.any():
                        target_payload["target_mean_S"] = float(np.mean(x_labeled_prior[sl]))
                        target_payload["target_std_S"] = float(np.std(x_labeled_prior[sl]))
                    if (~sl).any():
                        target_payload["target_mean_not_S"] = float(np.mean(x_labeled_prior[~sl]))
                        target_payload["target_std_not_S"] = float(np.std(x_labeled_prior[~sl]))
                    if sl.any() and (~sl).any():
                        target_payload["target_gap_S_minus_notS"] = (
                            target_payload["target_mean_S"] - target_payload["target_mean_not_S"]
                        )
                wandb.log(target_payload)
        except Exception:
            pass

        if policy == 'sl':
            # supervised learning policy
            # assume the initial predictions are the innate opinions
            platform_predictions[:n] = copy.deepcopy(x_labeled_prior)
            if model_name == 'perfect':
                platform_predictions[n:] = copy.deepcopy(x_unlabeled_prior)
            else:
                platform_predictions[n:] = predicting(model_name, X_features_labeled, x_labeled_prior, X_features_unlabeled,
                                                      df_labeled=df_labeled, df_unlabeled=df_unlabeled,
                                                      sim_params=sim_params)

        else:
            # steering policy
            x_prior = np.concatenate((x_labeled_prior, x_unlabeled_prior))
            platform_predictions = steering_params * steering_vector + np.diag(np.ones(agent_num) - steering_params) @ x_prior
        x_zero = (1.0 - platform_params) * x_star + platform_params * platform_predictions
        x_temp = copy.deepcopy(x_zero)
        # Precompute row-normalized weight matrix once per round (W/deg)
        normalized_w = weight_mat * degs_inv[:, None]
        one_minus_peer = 1.0 - peer_params
        for k in range(fj_steps):
            x_temp = peer_params * x_zero + one_minus_peer * (normalized_w @ x_temp)

        
        whole_record[:, t+1] = copy.deepcopy(x_temp)
        x_labeled_prior = copy.deepcopy(x_temp[:n])
        x_unlabeled_prior = copy.deepcopy(x_temp[n:])

        # Log population opinion stats per round
        try:
            import wandb
            if wandb.run is not None:
                dist_to_innate = float(np.linalg.norm(x_temp - x_star) / np.sqrt(agent_num))
                dist_to_uniform_mean = float(np.std(x_temp))  # 0 iff fully homogenized
                log_payload = {
                    "round": t + 1,
                    "opinion_mean": float(np.mean(x_temp)),
                    "opinion_std": float(np.std(x_temp)),
                    "opinion_min": float(np.min(x_temp)),
                    "opinion_max": float(np.max(x_temp)),
                    "opinion_q25": float(np.quantile(x_temp, 0.25)),
                    "opinion_q75": float(np.quantile(x_temp, 0.75)),
                    "dist_to_innate": dist_to_innate,
                    "opinion_range": float(np.max(x_temp) - np.min(x_temp)),
                    "opinion_hist": wandb.Histogram(x_temp, num_bins=40),
                    # Mean drift: how far the population has moved from innate
                    "mean_drift_from_innate": float(np.mean(x_temp) - x_star_mean),
                    "abs_mean_drift_from_innate": float(abs(np.mean(x_temp) - x_star_mean)),
                    "std_ratio_to_innate": float(np.std(x_temp) / max(x_star_std, 1e-9)),
                }
                # Real-time subgroup tracking. Adds nothing when no tracker is set.
                if _track_mask_full is not None:
                    mu_s = float(x_temp[_track_mask_full].mean())
                    mu_ns = float(x_temp[~_track_mask_full].mean())
                    # Four-way split: separates direct LM influence (unlabeled side)
                    # from peer-mediated spillover (labeled side anchors innate).
                    sl_mask = _track_mask_labeled
                    su_mask = _track_mask_unlabeled
                    x_lab = x_temp[:n]
                    x_unl = x_temp[n:]
                    mu_s_lab = float(x_lab[sl_mask].mean()) if sl_mask.any() else float("nan")
                    mu_ns_lab = float(x_lab[~sl_mask].mean()) if (~sl_mask).any() else float("nan")
                    mu_s_unl = float(x_unl[su_mask].mean()) if su_mask.any() else float("nan")
                    mu_ns_unl = float(x_unl[~su_mask].mean()) if (~su_mask).any() else float("nan")
                    log_payload.update({
                        "subgroup_tag": _tracker.tag,
                        "track_mu_S": mu_s,
                        "track_mu_not_S": mu_ns,
                        "track_gap_S_minus_notS": mu_s - mu_ns,
                        "track_abs_gap": abs(mu_s - mu_ns),
                        # Labeled side (anchored by innate via FJ).
                        "track_mu_S_labeled": mu_s_lab,
                        "track_mu_not_S_labeled": mu_ns_lab,
                        "track_gap_labeled": mu_s_lab - mu_ns_lab,
                        # Unlabeled side (driven directly by LM platform predictions).
                        "track_mu_S_unlabeled": mu_s_unl,
                        "track_mu_not_S_unlabeled": mu_ns_unl,
                        "track_gap_unlabeled": mu_s_unl - mu_ns_unl,
                    })
                wandb.log(log_payload)
        except Exception as e:
            print(f"[sim] wandb log skipped: {e}")

    return whole_record




def run_opinion_dynamics(innate_opinions, network_lcc, nodelist, model_name, X_features_labeled, X_features_unlabeled,
                         df_labeled=None, df_unlabeled=None):
    
    agent_num = len(innate_opinions)
    fj_K = 100
    retrain_T = int(os.environ.get("RETRAIN_T", 20))
    x_initial = copy.deepcopy(innate_opinions)

    # Init wandb for ALL model_name values (baselines too) so opinion_std gets logged
    run_tag_env = os.environ.get("RUN_TAG", "")
    try:
        import wandb
        if wandb.run is None:
            wandb.init(
                project="opinion-dynamics-llm",
                name=run_tag_env if run_tag_env else model_name,
                config={
                    "model_name": model_name,
                    "run_tag": run_tag_env,
                    "retrain_T": retrain_T,
                    "fj_K": fj_K,
                    "agent_num": agent_num,
                },
                reinit=False,
            )
        wandb.define_metric("round")
        for pat in ("opinion_*", "pred_*", "dist_to_innate", "opinion_range",
                    "mean_drift_from_innate", "abs_mean_drift_from_innate",
                    "std_ratio_to_innate", "target_*", "parse_fail_*"):
            wandb.define_metric(pat, step_metric="round")
    except Exception as e:
        print(f"[run_opinion_dynamics] wandb init skipped: {e}")

    param_folder = "pokec_dataset/parametric_params/"
    realworld_params = Path(param_folder)
    realworld_params.mkdir(exist_ok=True)

    # generate heterogeneous parameters
    platform_file_path = Path(param_folder + "hetero_platform_sus" + str(agent_num) + ".pkl")

    if not platform_file_path.exists():
        
        platform_sus = np.clip(np.random.normal(loc=0.9, scale=0.1, size=agent_num), 0.01, 0.99)
        with open(platform_file_path, "wb") as file:
            pickle.dump(platform_sus, file)
    else:
        with open(platform_file_path, "rb") as file:
            platform_sus = pickle.load(file)

    peer_file_path = Path(param_folder + "hetero_peer_sus" + str(agent_num) + ".pkl")

    if not peer_file_path.exists():

        peer_sus = np.clip(np.random.normal(loc=0.5, scale=0.1, size=agent_num), 0.01, 0.99)
        with open(peer_file_path, "wb") as file:
            pickle.dump(peer_sus, file)
    else:
        with open(peer_file_path, "rb") as file:
            peer_sus = pickle.load(file)
    # Apply LABELED_SPLIT permutation to peer_sus and platform_sus if set.
    _order = get_new_order(agent_num)
    peer_sus = np.asarray(peer_sus)[_order]
    platform_sus = np.asarray(platform_sus)[_order]


    steer_file_path = Path(param_folder + "hetero_steer_sus" + str(agent_num) + ".pkl")

    if not steer_file_path.exists():
        steer_strength = np.clip(np.random.normal(loc=0.1, scale=0.1, size=agent_num), 0.01, 0.99)
        with open(steer_file_path, "wb") as file:
            pickle.dump(steer_strength, file)

    else:
        with open(steer_file_path, "rb") as file:
            steer_strength = pickle.load(file)

    # for illustrating supervised learning policy, strong performativity
    platform_sus = np.ones(agent_num)
    steer_strength = np.zeros(agent_num)
    results_folder = "pokec_dataset/results/"
    os.makedirs(results_folder, exist_ok=True)
    run_tag = os.environ.get("RUN_TAG", "")   # append to cache filenames to avoid overwriting across configs
    # avoid "perfect_perfect"-style filenames when RUN_TAG equals model_name
    cache_key = model_name if (not run_tag or run_tag == model_name) else f"{model_name}_{run_tag}"
    if os.path.exists(results_folder + cache_key + "_equilibrium.pk"):
        with open(results_folder + cache_key + "_equilibrium.pk", "rb") as f:
            perform_equilibrium = pickle.load(f)
        with open(results_folder + cache_key + "_FJequilibrium.pk", "rb") as f:
            FJ_equilibrium = pickle.load(f)
    else:

        equilibrium_opinions = run_simulation(network=network_lcc, nodelist=nodelist, platform_params=platform_sus, 
                                            peer_params=peer_sus, steering_params=steer_strength, 
                                            steering_vector=None, fj_steps=fj_K, retrain_steps=retrain_T, 
                                            x_star=innate_opinions, policy='sl', model_name=model_name, 
                                            X_features_labeled=X_features_labeled, X_features_unlabeled=X_features_unlabeled,
                                            df_labeled=df_labeled, df_unlabeled=df_unlabeled)

        interval_num = retrain_T
        heatmap_res1 = np.zeros((agent_num, interval_num+1))
        heatmap_res1[:, 0] = copy.deepcopy(x_initial)
        for k in range(interval_num):
            heatmap_res1[:, k+1] = copy.deepcopy(equilibrium_opinions[:, (k+1)*int(equilibrium_opinions.shape[1]/interval_num)])


        x = np.arange(1, agent_num+1)
        FJ_equilibrium = heatmap_res1[:, 1]
        perform_equilibrium = heatmap_res1[:, -1]
        with open(results_folder + cache_key + "_equilibrium.pk", "wb") as f:
            pickle.dump(perform_equilibrium, f)
        with open(results_folder + cache_key + "_FJequilibrium.pk", "wb") as f:
            pickle.dump(FJ_equilibrium, f)
        # Full per-round trajectory: shape (agent_num, retrain_T+1), columns = rounds 0..T
        with open(results_folder + cache_key + "_trajectory.pk", "wb") as f:
            pickle.dump(heatmap_res1, f)

        # Subgroup mask sidecar: no-op when SFT_SUBGROUP is unset.
        # Row order of mask matches the (labeled-first, then unlabeled) row order
        # of heatmap_res1, so plot scripts can index trajectory rows directly.
        _sg = Subgroup.from_env()
        if _sg is not None and df_labeled is not None and df_unlabeled is not None:
            _df_full = pd.concat(
                [df_labeled.reset_index(drop=True),
                 df_unlabeled.reset_index(drop=True)],
                ignore_index=True,
            )
            _sg.save_sidecar(results_folder, cache_key, _df_full)
        
    
    # reference with perfect predictions are pre-generated
    if model_name == 'perfect':
        with open("pokec_dataset/results/perfect_equilibrium.pk", "wb") as f:
            pickle.dump(perform_equilibrium, f)
        perfect_equilibrium = copy.deepcopy(perform_equilibrium)
    else:
        if os.path.exists("pokec_dataset/results/perfect_equilibrium.pk"):
            with open("pokec_dataset/results/perfect_equilibrium.pk", "rb") as f:
                perfect_equilibrium = pickle.load(f)
        else:
            print("Reference missing! Please generate with policy== perfect first!")
            perfect_equilibrium = np.full(agent_num, np.nan)



    innate_mean = innate_opinions.mean(axis=0)
    innate_std = innate_opinions.std(axis=0)

    fj_mean = FJ_equilibrium.mean(axis=0)
    fj_std = FJ_equilibrium.std(axis=0)

    performative_mean = perform_equilibrium.mean(axis=0)
    performative_std = perform_equilibrium.std(axis=0)

    perfect_mean = perfect_equilibrium.mean(axis=0)
    perfect_std = perfect_equilibrium.std(axis=0)

    
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    x = np.arange(1, agent_num+1)

    plt.scatter(x, innate_opinions, s=5, color=colors[0], label=r"$x^*$")
    plt.scatter(x, FJ_equilibrium, s=5, color=colors[1], label=r"FJ($x^*$)")
    plt.scatter(x, perfect_equilibrium, s=5, color=colors[3], label=r"$x_{PS}$ (perfect)")
    plt.scatter(x, perform_equilibrium, s=5, color=colors[2], label=r"$x_{PS}$ (imperfect)")


    plt.fill_between(x, innate_mean - innate_std, innate_mean + innate_std, color=colors[0], alpha=0.2)
    plt.fill_between(x, fj_mean - fj_std, fj_mean + fj_std, color=colors[1], alpha=0.2)
    plt.fill_between(x, performative_mean - performative_std, performative_mean + performative_std, color=colors[2], alpha=0.2)
    plt.fill_between(x, perfect_mean - perfect_std, perfect_mean + perfect_std, color=colors[3], alpha=0.2)
    
    
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    plt.ylabel("Opinions", fontsize=13)
    plt.legend(loc="upper left", bbox_to_anchor=(1,1), frameon=False, fontsize=10)
    plt.savefig(param_folder + model_name + "_parametric_sl.pdf", bbox_inches='tight')


def plot_adjust(model_name):
    results_folder = "pokec_dataset/results/"
    param_folder = "pokec_dataset/parametric_params/"

    with open(results_folder + model_name + "_equilibrium.pk", "rb") as f:
        perform_equilibrium = pickle.load(f)
    with open(results_folder + model_name + "_FJequilibrium.pk", "rb") as f:
        FJ_equilibrium = pickle.load(f)
    with open("pokec_dataset/parametric_params/y_label2163.pk", "rb") as f:
        y_label = pickle.load(f)
    with open("pokec_dataset/parametric_params/y_unlabel_label2163.pk", "rb") as f:
        y_unlabel_label = pickle.load(f)
    innate_opinions = np.array(y_label + y_unlabel_label)
    with open("pokec_dataset/results/perfect_equilibrium.pk", "rb") as f:
        perfect_equilibrium = pickle.load(f)
    innate_mean = innate_opinions.mean(axis=0)
    innate_std = innate_opinions.std(axis=0)

    fj_mean = FJ_equilibrium.mean(axis=0)
    fj_std = FJ_equilibrium.std(axis=0)

    performative_mean = perform_equilibrium.mean(axis=0)
    performative_std = perform_equilibrium.std(axis=0)

    perfect_mean = perfect_equilibrium.mean(axis=0)
    perfect_std = perfect_equilibrium.std(axis=0)

    
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    x = np.arange(1, 2164)

    plt.scatter(x, innate_opinions, s=4, color=colors[0], label=r"$x^*$")
    plt.scatter(x, FJ_equilibrium, s=4, color=colors[1], label=r"FJ($x^*)$")
    plt.scatter(x, perfect_equilibrium, s=4, color=colors[3], label=r"$x_{PS}$ (perfect)")
    plt.scatter(x, perform_equilibrium, s=4, color=colors[2], label=r"$x_{PS}$ (imperfect)")

    x_right = x.max() + 45

    plt.errorbar([x_right+210], [innate_mean], yerr=[innate_std], fmt="o", markersize=4, capsize=4, color=colors[0])
    plt.errorbar([x_right+140], [fj_mean], yerr=[fj_std], fmt="o", capsize=4, markersize=4, color=colors[1])
    plt.errorbar([x_right+70], [perfect_mean], yerr=[perfect_std], fmt="o", markersize=4, capsize=4, color=colors[3])
    plt.errorbar([x_right], [performative_mean], yerr=[performative_std], fmt="o", markersize=4, capsize=4, color=colors[2])

    
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    plt.ylabel("Opinions", fontsize=18)
    plt.gca().set_xticklabels([])
    plt.yticks(fontsize=12)
    plt.legend(loc="upper left", bbox_to_anchor=(0.97,1), frameon=False, fontsize=15)
    plt.savefig(param_folder + model_name + "_parametric_sl.pdf", bbox_inches='tight')

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
    
    # Step 1: compute or load y_label, y_unlabel_label from the POSITIONAL split.
    # These cached files contain innate scores for the first-80% / last-20%
    # row partition. We load them in positional form, then permute below.
    n_pos = int(len(df) * 0.8)
    if not os.path.exists("pokec_dataset/parametric_params/y_label" + str(len(df)) + ".pk"):
        df_labeled_pos = df.iloc[:n_pos].copy()
        df_unlabeled_pos = df.iloc[n_pos:].copy()
        y_label_pos = compute_score(df_labeled_pos, target_column, args)
        y_unlabel_label_pos = compute_score(df_unlabeled_pos, target_column, args)
        with open("pokec_dataset/parametric_params/y_label" + str(len(df)) + ".pk", "wb") as f:
            pickle.dump(y_label_pos, f)
        with open("pokec_dataset/parametric_params/y_unlabel_label" + str(len(df)) + ".pk", "wb") as f:
            pickle.dump(y_unlabel_label_pos, f)
    else:
        with open("pokec_dataset/parametric_params/y_label" + str(len(df)) + ".pk", "rb") as f:
            y_label_pos = pickle.load(f)
        with open("pokec_dataset/parametric_params/y_unlabel_label" + str(len(df)) + ".pk", "rb") as f:
            y_unlabel_label_pos = pickle.load(f)

    # Step 2: apply LABELED_SPLIT permutation (identity if env var unset).
    # The permutation puts the active labeled subset at rows [0, n).
    _order = get_new_order(len(df))
    df = df.iloc[_order].reset_index(drop=True)
    _innate_pos = np.array(list(y_label_pos) + list(y_unlabel_label_pos), dtype=np.float64)
    _innate_new = _innate_pos[_order]
    n = get_n_labeled(len(df))
    y_label = list(_innate_new[:n])
    y_unlabel_label = list(_innate_new[n:])
    df_labeled = df.iloc[:n].copy()
    df_unlabeled = df.iloc[n:].copy()

    # extract features from mixed data types
    filtered_textual_features = [text for text in textual_features if text != target_column]
    df_labeled[numerical_features_extended] = df_labeled[numerical_features_extended].apply(pd.to_numeric, errors="coerce")
    df_labeled[categorical_features] = df_labeled[categorical_features].astype(str)
    df_labeled[filtered_textual_features] = df_labeled[filtered_textual_features].astype(str)
    df_unlabeled[numerical_features_extended] = df_unlabeled[numerical_features_extended].apply(pd.to_numeric, errors="coerce")
    df_unlabeled[categorical_features] = df_unlabeled[categorical_features].astype(str)
    df_unlabeled[filtered_textual_features] = df_unlabeled[filtered_textual_features].astype(str)

    # Split-aware cache filenames for the feature matrices: under LABELED_SPLIT,
    # df_labeled/df_unlabeled are different rows, so we must NOT reuse the
    # positional-split cached features.
    _suffix = split_suffix()
    _lab_feat = ("pokec_dataset/labeled_feature_matrix_" + target_column + _suffix
                 + "_" + str(include_graph_features) + ".pk")
    _unl_feat = ("pokec_dataset/unlabeled_feature_matrix_" + target_column + _suffix
                 + "_" + str(include_graph_features) + ".pk")
    if os.path.exists(_lab_feat):
        with open(_lab_feat, "rb") as f:
            X_features_labeled = pickle.load(f)
        print("Feature matrix shape:", X_features_labeled.shape)
        with open(_unl_feat, "rb") as f:
            X_features_unlabeled = pickle.load(f)
    else:
        X_features_labeled = extract_features(df_labeled, args, filtered_textual_features, numerical_features_extended)
        X_features_unlabeled = extract_features(df_unlabeled, args, filtered_textual_features, numerical_features_extended)
        with open(_lab_feat, "wb") as f:
            pickle.dump(X_features_labeled, f)
        with open(_unl_feat, "wb") as f:
            pickle.dump(X_features_unlabeled, f)
    
    model_name = os.environ.get("MODEL_NAME", "llm")  # "neural_net" | "ridge" | "mean" | "perfect" | "llm"
    # computed sentiment scores are assumed to be innate opinions, x_star
    innate_opinions = np.array(y_label + y_unlabel_label)
    adjust_plot = False 
    if adjust_plot:
        plot_adjust(model_name)
    else: 
        run_opinion_dynamics(innate_opinions, network_lcc, df["user_id"].values, model_name, X_features_labeled, X_features_unlabeled,
                             df_labeled=df_labeled, df_unlabeled=df_unlabeled)



if __name__ == "__main__":
    main()