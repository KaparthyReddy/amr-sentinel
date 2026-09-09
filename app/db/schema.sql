CREATE TABLE surveillance_logs (
    id BIGSERIAL PRIMARY KEY,
    sequence_hash VARCHAR(64) NOT NULL,
    sequence_length INTEGER NOT NULL,
    predicted_mechanism VARCHAR(100) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    novelty_distance DOUBLE PRECISION NOT NULL,
    is_novel BOOLEAN NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX idx_surveillance_logs_mechanism ON surveillance_logs(predicted_mechanism);
CREATE INDEX idx_surveillance_logs_is_novel ON surveillance_logs(is_novel);
CREATE INDEX idx_surveillance_logs_created_at ON surveillance_logs(created_at);
