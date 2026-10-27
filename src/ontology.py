"""Load ontology.yaml and check DataFrames against it.

The ontology is the contract; the graph builder must obey it. Checks:
  * every relation connects entities that exist in the ontology
  * node/edge tables have exactly the declared columns, with the declared types
  * every edge points at nodes that exist (no dangling relationships)
  * every entity key is unique
"""
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config import PROJECT_ROOT

TYPE_MAP = {"string": "string", "long": "bigint", "int": "int", "double": "double",
            "boolean": "boolean", "timestamp": "timestamp"}


@dataclass
class Entity:
    name: str
    key: str
    description: str
    properties: dict[str, str] = field(default_factory=dict)


@dataclass
class Relation:
    name: str
    source: str
    target: str
    description: str
    derived: bool = False
    properties: dict[str, str] = field(default_factory=dict)


@dataclass
class Ontology:
    entities: dict[str, Entity]
    relations: dict[str, Relation]

    @classmethod
    def load(cls, path: Path = PROJECT_ROOT / "ontology.yaml") -> "Ontology":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        entities = {
            name: Entity(name, e["key"], e.get("description", ""), e.get("properties") or {})
            for name, e in raw["entities"].items()
        }
        relations = {
            name: Relation(name, r["from"], r["to"], r.get("description", "").strip(),
                           r.get("derived", False), r.get("properties") or {})
            for name, r in raw["relations"].items()
        }
        onto = cls(entities, relations)
        onto._check_self()
        return onto

    def _check_self(self) -> None:
        for r in self.relations.values():
            for end in (r.source, r.target):
                if end not in self.entities:
                    raise ValueError(f"Relation {r.name} refers to unknown entity '{end}'")
        for e in self.entities.values():
            for t in e.properties.values():
                if t not in TYPE_MAP:
                    raise ValueError(f"Entity {e.name}: unknown type '{t}'")

    # ---------------- checks against data ----------------
    @staticmethod
    def _check_columns(df: DataFrame, expected: dict[str, str], what: str) -> list[str]:
        actual = dict(df.dtypes)
        errors = []
        if set(actual) != set(expected):
            errors.append(f"{what}: columns {sorted(actual)} != ontology {sorted(expected)}")
        for col, typ in expected.items():
            if col in actual and actual[col] != TYPE_MAP[typ]:
                errors.append(f"{what}.{col}: type {actual[col]} != ontology {typ}")
        return errors

    def node_columns(self, entity: str) -> dict[str, str]:
        e = self.entities[entity]
        return {"id": "string", **e.properties}

    def edge_columns(self, relation: str) -> dict[str, str]:
        return {"src": "string", "dst": "string", **self.relations[relation].properties}

    def validate(self, nodes: dict[str, DataFrame], edges: dict[str, DataFrame]) -> list[str]:
        """Return a list of rule violations (empty list = the graph obeys the ontology)."""
        errors = []
        for name in self.entities:
            if name not in nodes:
                errors.append(f"missing node table for entity {name}")
                continue
            errors += self._check_columns(nodes[name], self.node_columns(name), f"node {name}")
            dupes = nodes[name].groupBy("id").count().filter("count > 1").count()
            if dupes:
                errors.append(f"node {name}: {dupes:,} duplicate ids")

        for name, rel in self.relations.items():
            if name not in edges:
                errors.append(f"missing edge table for relation {name}")
                continue
            df = edges[name]
            errors += self._check_columns(df, self.edge_columns(name), f"edge {name}")
            for side, entity in (("src", rel.source), ("dst", rel.target)):
                if entity in nodes:
                    ids = nodes[entity].select(F.col("id").alias("_node_id"))
                    dangling = df.join(ids, F.col(side) == F.col("_node_id"), "left_anti").count()
                    if dangling:
                        errors.append(f"edge {name}: {dangling:,} rows whose {side} is not a {entity}")
        return errors

