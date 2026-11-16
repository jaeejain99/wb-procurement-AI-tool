"""Table store with Delta semantics. The pipeline is written once against `Lake`:

  LocalLake  - pickled pandas frames in a folder (used by tests / laptops; fully exercised here)
  DeltaLake  - Unity Catalog Delta tables through Spark (Databricks). NOT run in the build sandbox (no Spark/workspace);
               the surface is deliberately tiny so there is little to get wrong: write / merge / read / enable_cdf.

Medallion layout (table names):
  bronze_contracts bronze_projects bronze_governance bronze_disbursements bronze_isr bronze_docs   raw, append-only, with _ingest_ts
  silver_chunks                                                                                      parsed + chunked documents
  gold_contract_features gold_chunks gold_kg_edges gold_conflicts gold_alerts                         serving / ML / index sources
"""
from __future__ import annotations
import os, pickle
import pandas as pd


class Lake:
    def write(self, name, df, mode="overwrite"): raise NotImplementedError
    def merge(self, name, df, keys): raise NotImplementedError
    def read(self, name) -> pd.DataFrame: raise NotImplementedError
    def exists(self, name) -> bool: raise NotImplementedError
    def delete_in(self, name, col, values): raise NotImplementedError
    def enable_cdf(self, name): pass


class LocalLake(Lake):
    def __init__(self, root):
        self.root = root; os.makedirs(root, exist_ok=True)

    def _p(self, name): return os.path.join(self.root, f"{name}.pkl")

    def exists(self, name): return os.path.exists(self._p(name))

    def read(self, name) -> pd.DataFrame:
        with open(self._p(name), "rb") as f:
            return pickle.load(f)

    def write(self, name, df, mode="overwrite"):
        if mode == "append" and self.exists(name):
            df = pd.concat([self.read(name), df], ignore_index=True)
        df = df.copy(); df.attrs = {}                # plain pandas only: no producer objects (e.g. Spark metrics) in the pickle
        with open(self._p(name), "wb") as f:
            pickle.dump(df, f)

    def delete_in(self, name, col, values):
        df = self.read(name); self.write(name, df[~df[col].isin(list(values))].reset_index(drop=True))

    def merge(self, name, df, keys):               # upsert: matched -> update, not matched -> insert
        if self.exists(name):
            df = pd.concat([self.read(name), df], ignore_index=True).drop_duplicates(keys, keep="last")
        self.write(name, df.reset_index(drop=True))


class DeltaLake(Lake):
    """Unity Catalog tables: <catalog>.<schema>.<name>."""

    def __init__(self, spark, catalog, schema):
        self.spark, self.catalog, self.schema = spark, catalog, schema

    def fq(self, name): return f"`{self.catalog}`.`{self.schema}`.`{name}`"

    def exists(self, name): return self.spark.catalog.tableExists(f"{self.catalog}.{self.schema}.{name}")

    def read(self, name) -> pd.DataFrame:
        df = self.spark.table(self.fq(name)).toPandas()
        df.attrs = {}               # Spark Connect puts query-metric objects in .attrs; they would drag pyspark into pickles
        return df

    def _sdf(self, df):
        """Explicit schema from pandas dtypes: Spark cannot infer types for empty tables (disbursements) or all-null columns."""
        from pyspark.sql import types as T
        api, fields, cols = pd.api.types, [], []
        for c in df.columns:
            s = df[c]
            nn = s.dropna()
            if api.is_bool_dtype(s):
                t = T.BooleanType()
            elif api.is_integer_dtype(s):
                t = T.LongType()
            elif api.is_float_dtype(s):
                t = T.DoubleType()
            elif api.is_datetime64_any_dtype(s) or (len(nn) and all(isinstance(v, pd.Timestamp) for v in nn)):
                t = T.TimestampType(); s = pd.to_datetime(s)
            else:
                t = T.StringType(); s = s.map(lambda v: None if v is None or (isinstance(v, float) and v != v) else str(v))
            fields.append(T.StructField(str(c), t, True)); cols.append(s)

        def py(v):
            if v is None or v is pd.NaT or (isinstance(v, float) and v != v):
                return None
            return v.to_pydatetime() if isinstance(v, pd.Timestamp) else v.item() if hasattr(v, "item") else v
        rows = [tuple(py(v) for v in r) for r in zip(*cols)] if cols else []
        return self.spark.createDataFrame(rows, T.StructType(fields))

    def write(self, name, df, mode="overwrite"):
        w = self._sdf(df).write.format("delta").mode(mode)
        (w.option("overwriteSchema", "true") if mode == "overwrite" else w.option("mergeSchema", "true")).saveAsTable(self.fq(name))

    def merge(self, name, df, keys):
        if not self.exists(name):
            return self.write(name, df)
        from delta.tables import DeltaTable
        cond = " AND ".join(f"t.`{k}` = s.`{k}`" for k in keys)
        (DeltaTable.forName(self.spark, self.fq(name)).alias("t").merge(self._sdf(df).alias("s"), cond)
         .whenMatchedUpdateAll().whenNotMatchedInsertAll().execute())

    def delete_in(self, name, col, values):
        from delta.tables import DeltaTable
        from pyspark.sql import functions as F
        DeltaTable.forName(self.spark, self.fq(name)).delete(F.col(col).isin(list(values)))

    def enable_cdf(self, name):                    # required by Vector Search delta-sync indexes
        self.spark.sql(f"ALTER TABLE {self.fq(name)} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
