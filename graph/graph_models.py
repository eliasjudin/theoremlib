"""
Graph Models Module

Handles the construction and analysis of theorem dependency graphs,
supporting multiple edge types and graph-based operations.
"""

import networkx as nx
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import json
from dataclasses import dataclass

class EdgeType(str, Enum):
    LOGICAL_DEPENDENCY = "logical_dependency"
    PROOF_TECHNIQUE = "proof_technique"
    EQUIVALENCE = "equivalence"

@dataclass
class GraphEdgeMetadata:
    confidence: float
    description: Optional[str] = None
    references: List[str] = None

class TheoremGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.edge_metadata: Dict[Tuple[str, str, str], GraphEdgeMetadata] = {}

    def add_theorem_node(
        self,
        theorem_id: str,
        metadata: Dict
    ) -> None:
        """Add a theorem node to the graph with its metadata"""
        self.graph.add_node(theorem_id, **metadata)

    def add_edge(
        self,
        from_theorem: str,
        to_theorem: str,
        edge_type: EdgeType,
        metadata: GraphEdgeMetadata
    ) -> None:
        """Add an edge between theorems with type and metadata"""
        self.graph.add_edge(
            from_theorem,
            to_theorem,
            key=edge_type,
            weight=metadata.confidence
        )
        self.edge_metadata[(from_theorem, to_theorem, edge_type)] = metadata

    def get_dependencies(
        self,
        theorem_id: str,
        depth: int = 1,
        edge_types: Optional[Set[EdgeType]] = None
    ) -> nx.DiGraph:
        """Get a subgraph of dependencies up to specified depth"""
        if edge_types is None:
            edge_types = set(EdgeType)

        def edge_filter(u, v, k, d):
            return k in edge_types

        subgraph = nx.ego_graph(
            self.graph,
            theorem_id,
            radius=depth,
            edge_filter=edge_filter
        )
        return subgraph

    def find_proof_technique_clusters(self) -> List[Set[str]]:
        """Find clusters of theorems sharing similar proof techniques"""
        technique_graph = nx.Graph()

        # Add edges between theorems with shared proof techniques
        for u, v, k, d in self.graph.edges(keys=True, data=True):
            if k == EdgeType.PROOF_TECHNIQUE:
                technique_graph.add_edge(u, v, weight=d['weight'])

        # Find communities using Louvain method
        try:
            import community
            return list(community.best_partition(technique_graph).values())
        except ImportError:
            # Fallback to connected components if python-louvain is not available
            return list(nx.connected_components(technique_graph))

    def find_equivalent_theorems(self, theorem_id: str) -> Set[str]:
        """Find all theorems equivalent to the given theorem"""
        equiv_edges = [(u, v) for u, v, k in self.graph.edges(keys=True)
                      if k == EdgeType.EQUIVALENCE]
        equiv_graph = nx.Graph(equiv_edges)
        
        if theorem_id in equiv_graph:
            return set(nx.node_connected_component(equiv_graph, theorem_id))
        return {theorem_id}

    def get_prerequisite_chain(self, theorem_id: str) -> List[List[str]]:
        """Get all prerequisite chains leading to the theorem"""
        paths = []
        for node in self.graph.nodes():
            if node != theorem_id:
                for path in nx.all_simple_paths(
                    self.graph,
                    node,
                    theorem_id,
                    cutoff=None
                ):
                    if all(self.graph.get_edge_data(path[i], path[i+1])[0]['key'] 
                          == EdgeType.LOGICAL_DEPENDENCY
                          for i in range(len(path)-1)):
                        paths.append(path)
        return paths

    def calculate_theorem_impact(self, theorem_id: str) -> float:
        """Calculate the impact score of a theorem based on its dependencies"""
        dependent_theorems = set()
        for path in nx.single_source_shortest_path(self.graph, theorem_id).values():
            dependent_theorems.update(path)
        
        impact_score = len(dependent_theorems)
        
        # Weight by the average confidence of dependencies
        if impact_score > 1:  # More than just the theorem itself
            confidence_sum = sum(
                self.edge_metadata.get((u, v, k), GraphEdgeMetadata(confidence=0.5)).confidence
                for u, v, k in self.graph.edges(keys=True)
                if u == theorem_id
            )
            avg_confidence = confidence_sum / self.graph.out_degree(theorem_id)
            impact_score *= avg_confidence
            
        return impact_score

    def to_json(self) -> str:
        """Export the graph to JSON format"""
        return json.dumps({
            "nodes": [
                {
                    "id": node,
                    "metadata": data
                }
                for node, data in self.graph.nodes(data=True)
            ],
            "edges": [
                {
                    "from": u,
                    "to": v,
                    "type": k,
                    "metadata": self.edge_metadata.get((u, v, k), 
                                                     GraphEdgeMetadata(confidence=0.5)).__dict__
                }
                for u, v, k in self.graph.edges(keys=True)
            ]
        })

    @classmethod
    def from_json(cls, json_str: str) -> 'TheoremGraph':
        """Create a graph from JSON format"""
        data = json.loads(json_str)
        graph = cls()
        
        for node in data["nodes"]:
            graph.add_theorem_node(node["id"], node["metadata"])
            
        for edge in data["edges"]:
            graph.add_edge(
                edge["from"],
                edge["to"],
                EdgeType(edge["type"]),
                GraphEdgeMetadata(**edge["metadata"])
            )
            
        return graph