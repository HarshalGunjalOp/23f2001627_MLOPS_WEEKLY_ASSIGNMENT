from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, String

# 1. Define the Data Source (Offline)
iris_source = FileSource(
    path="data/iris_dataset.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# 2. Define the Entity (The ID)
iris = Entity(
    name="iris_id",
    join_keys=["iris_id"],
    value_type=ValueType.INT64,
    description="Identifier for specific iris plant",
)

# 3. Define the Feature View
iris_feature_view = FeatureView(
    name="iris_features",
    entities=[iris],
    ttl=timedelta(days=3650), # 10-year TTL so old data isn't rejected
    source=iris_source,
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
        Field(name="species", dtype=String),
    ],
)