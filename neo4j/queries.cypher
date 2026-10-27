// =====================================================================
// Questions to ask the knowledge graph in Neo4j Browser (http://localhost:7474)
// Run one block at a time.
// =====================================================================

// 1. See the ontology itself: which labels connect to which, and how
CALL db.schema.visualization();

// 2. The category taxonomy as a tree
MATCH path = (:Category)-[:SUBCATEGORY_OF*]->(top:Category {level: 1})
RETURN path;

// 3. Brand switching: shoppers who viewed Samsung but bought another brand instead
MATCH (:Brand {id: 'samsung'})-[s:SWITCHED_TO]->(b:Brand)
RETURN b.id AS bought_instead, s.category_main AS category, s.sessions AS sessions
ORDER BY sessions DESC;

// 4. Same question answered from raw behaviour, not the pre-computed edge:
//    users who viewed a Samsung product and bought a non-Samsung product
MATCH (u:User)-[:HAD]->(s1:Session)-[:VIEWED]->(:Product)-[:MADE_BY]->(:Brand {id: 'samsung'}),
      (u)-[:HAD]->(s2:Session)-[:PURCHASED]->(:Product)-[:MADE_BY]->(b:Brand)
WHERE b.id <> 'samsung'
RETURN b.id AS brand, count(DISTINCT u) AS customers
ORDER BY customers DESC LIMIT 10;

// 5. The most central products in the co-purchase network (PageRank)
MATCH (p:Product)
OPTIONAL MATCH (p)-[:MADE_BY]->(b:Brand)
OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Category)
RETURN p.id, b.id AS brand, c.id AS category, p.pagerank
ORDER BY p.pagerank DESC LIMIT 10;

// 6. A product's neighbourhood: what it is bought with (draw it as a graph)
MATCH (p:Product)
WITH p ORDER BY p.pagerank DESC LIMIT 1
MATCH path = (p)-[r:BOUGHT_TOGETHER]->(:Product)
WHERE r.lift > 1
RETURN path LIMIT 25;

// 7. Explainable recommendation for one user:
//    products bought together with what they bought, that they don't own yet
MATCH (u:User)-[:HAD]->(:Session)-[:PURCHASED]->(owned:Product)-[r:BOUGHT_TOGETHER]->(rec:Product)
WHERE u.id = $userId                       // set with  :param userId => '512017816'
  AND r.lift > 1
  AND NOT EXISTS { (u)-[:HAD]->(:Session)-[:PURCHASED]->(rec) }
RETURN rec.id AS recommend, sum(r.confidence) AS score,
       collect(owned.id + ' (' + r.customers + ' shared buyers)')[0..3] AS because_you_bought
ORDER BY score DESC LIMIT 10;

// 8. Abandoned carts: carted but never purchased by that user, by category
MATCH (u:User)-[:HAD]->(:Session)-[:CARTED]->(p:Product)-[:BELONGS_TO]->(c:Category)
WHERE NOT EXISTS { (u)-[:HAD]->(:Session)-[:PURCHASED]->(p) }
RETURN c.id AS category, count(*) AS abandoned
ORDER BY abandoned DESC;
