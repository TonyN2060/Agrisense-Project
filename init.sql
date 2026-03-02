-- ==============================================================================
-- AgriSense ML Module - PostgreSQL Database Initialization
-- ==============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ==============================================================================
-- Tenants & Authentication
-- ==============================================================================

CREATE TABLE IF NOT EXISTS tenants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    api_key_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_tenants_api_key ON tenants(api_key_hash);

-- ==============================================================================
-- Cold Rooms
-- ==============================================================================

CREATE TABLE IF NOT EXISTS cold_rooms (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    location VARCHAR(255),
    volume_m3 DECIMAL(10, 2),
    max_capacity_kg DECIMAL(10, 2),
    crop_type VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_cold_rooms_tenant ON cold_rooms(tenant_id);

-- ==============================================================================
-- Sensor Readings (Time-Series)
-- ==============================================================================

CREATE TABLE IF NOT EXISTS sensor_readings (
    id BIGSERIAL PRIMARY KEY,
    cold_room_id UUID REFERENCES cold_rooms(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    temperature DECIMAL(5, 2),
    humidity DECIMAL(5, 2),
    co2_ppm DECIMAL(10, 2),
    light_lux DECIMAL(10, 2),
    door_open BOOLEAN DEFAULT FALSE,
    compressor_duty DECIMAL(5, 2)
);

CREATE INDEX idx_sensor_readings_room_time ON sensor_readings(cold_room_id, timestamp DESC);

-- Partition by month for better performance (PostgreSQL 12+)
-- CREATE TABLE sensor_readings_2024_01 PARTITION OF sensor_readings
--     FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- ==============================================================================
-- Predictions
-- ==============================================================================

CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    cold_room_id UUID REFERENCES cold_rooms(id) ON DELETE CASCADE,
    sensor_reading_id BIGINT REFERENCES sensor_readings(id),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    prediction VARCHAR(20) NOT NULL,
    prediction_class SMALLINT NOT NULL,
    confidence DECIMAL(4, 3),
    confidence_lower DECIMAL(4, 3),
    confidence_upper DECIMAL(4, 3),
    rsl_hours DECIMAL(10, 2),
    model_version VARCHAR(50),
    processing_time_ms DECIMAL(8, 3)
);

CREATE INDEX idx_predictions_room_time ON predictions(cold_room_id, timestamp DESC);
CREATE INDEX idx_predictions_class ON predictions(prediction_class);

-- ==============================================================================
-- Model Metrics (for drift detection)
-- ==============================================================================

CREATE TABLE IF NOT EXISTS model_metrics (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    model_version VARCHAR(50) NOT NULL,
    accuracy DECIMAL(5, 4),
    f1_score DECIMAL(5, 4),
    precision_score DECIMAL(5, 4),
    recall_score DECIMAL(5, 4),
    samples_evaluated INT,
    drift_detected BOOLEAN DEFAULT FALSE,
    kl_divergence DECIMAL(8, 6)
);

CREATE INDEX idx_model_metrics_version ON model_metrics(model_version, timestamp DESC);

-- ==============================================================================
-- Alert Events
-- ==============================================================================

CREATE TABLE IF NOT EXISTS alerts (
    id BIGSERIAL PRIMARY KEY,
    cold_room_id UUID REFERENCES cold_rooms(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,  -- INFO, WARNING, CRITICAL
    message TEXT,
    prediction_id BIGINT REFERENCES predictions(id),
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(255),
    acknowledged_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_alerts_room_time ON alerts(cold_room_id, timestamp DESC);
CREATE INDEX idx_alerts_severity ON alerts(severity) WHERE acknowledged = FALSE;

-- ==============================================================================
-- Audit Log
-- ==============================================================================

CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tenant_id UUID REFERENCES tenants(id),
    user_id VARCHAR(255),
    action VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(255),
    details JSONB
);

CREATE INDEX idx_audit_log_tenant ON audit_log(tenant_id, timestamp DESC);

-- ==============================================================================
-- Views
-- ==============================================================================

-- Latest readings per cold room
CREATE OR REPLACE VIEW v_latest_readings AS
SELECT DISTINCT ON (cold_room_id)
    cold_room_id,
    timestamp,
    temperature,
    humidity,
    co2_ppm
FROM sensor_readings
ORDER BY cold_room_id, timestamp DESC;

-- Prediction summary by cold room
CREATE OR REPLACE VIEW v_prediction_summary AS
SELECT
    cold_room_id,
    DATE_TRUNC('hour', timestamp) AS hour,
    COUNT(*) AS prediction_count,
    AVG(confidence) AS avg_confidence,
    COUNT(*) FILTER (WHERE prediction_class >= 3) AS critical_count
FROM predictions
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY cold_room_id, DATE_TRUNC('hour', timestamp);

-- ==============================================================================
-- Functions
-- ==============================================================================

-- Function to clean old sensor readings (retention policy)
CREATE OR REPLACE FUNCTION cleanup_old_readings(retention_days INT DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM sensor_readings
    WHERE timestamp < NOW() - (retention_days || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ==============================================================================
-- Initial Data
-- ==============================================================================

-- Insert default tenant for testing
INSERT INTO tenants (id, name, api_key_hash)
VALUES (
    'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11',
    'Demo Tenant',
    'demo-key-hash'
) ON CONFLICT DO NOTHING;

-- Insert demo cold room
INSERT INTO cold_rooms (id, tenant_id, name, location, crop_type)
VALUES (
    'b0eebc99-9c0b-4ef8-bb6d-6bb9bd380a22',
    'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11',
    'Demo Cold Room',
    'Warehouse A',
    'avocado'
) ON CONFLICT DO NOTHING;

-- ==============================================================================
-- Grants (adjust as needed)
-- ==============================================================================

-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO agrisense_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO agrisense_app;
