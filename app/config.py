from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql://amr_user:change_me_locally@localhost:5434/amr_sentinel"
    esm_model_name: str = "esm2_t6_8M_UR50D"
    novelty_distance_threshold: float = 0.35
    app_port: int = 8000

    classifier_path: str = "data/processed/classifier.joblib"
    label_encoder_path: str = "data/processed/label_encoder.joblib"
    centroids_path: str = "data/processed/class_centroids.joblib"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())


settings = Settings()
