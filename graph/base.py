"""
Graph Base Module

Provides core graph operations and utilities for theorem dependency graph construction.
"""

from typing import Dict, List, Set, Optional, Any, Tuple, TypeVar, Generic
import logging
from enum import Enum
from dataclasses import dataclass
from functools import wraps
import signal
from contextlib import contextmanager
import numpy as np
import sys
from importlib import util

from .graph_models import (
    EdgeType, GraphNodeMetadata, GraphEdgeMetadata,
    GraphNode, GraphEdge, HAVE_NETWORKX, HAVE_COMMUNITY
)

logger = logging.getLogger(__name__)

# Import networkx conditionally
nx = None
if util.find_spec("networkx"):
    import networkx as nx

# Define type variable for graph type
T = TypeVar('T')

# Custom timeout implementation
class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds: int):
    """Context manager for timing out long operations"""
    def signal_handler(signum, frame):
        raise TimeoutError("Operation timed out")

    # Register signal handler on Unix-like systems
    if sys.platform != 'win32':
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)

    try:
        yield
    finally:
        if sys.platform != 'win32':
            signal.alarm(0)

@dataclass
class GraphStats:
    """Statistics about the theorem graph"""
    node_count: int
    edge_count: int
    density: float
    avg_degree: float
    clustering_coefficient: float
    connected_components: int
    avg_path_length: float
    diameter: Optional[float]
    centrality_scores: Dict[str, float]

class GraphAnalyzer(Generic[T]):
    """Analyzer class for theorem graph metrics and properties"""
    
    def __init__(self, graph: T):
        if not HAVE_NETWORKX or nx is None:
            raise ImportError(
                "networkx is required for graph analysis. "
                "Install it with: pip install networkx"
            )
        if not isinstance(graph, nx.MultiDiGraph):
            raise ValueError("graph must be a networkx.MultiDiGraph instance")
        self.graph = graph
        self.nx = nx  # Store reference to avoid repeated lookups

    def compute_graph_stats(self) -> Optional[GraphStats]:
        """Compute comprehensive graph statistics"""
        if not HAVE_NETWORKX or self.nx is None:
            logger.error("networkx required for graph statistics")
            return None

        try:
            with timeout(30):  # 30 second timeout
                nx = self.nx  # Local reference for better readability
                
                # Basic metrics
                node_count = self.graph.number_of_nodes()
                edge_count = self.graph.number_of_edges()
                density = nx.density(self.graph)
                
                # Degree statistics
                degrees = [d for _, d in self.graph.degree()]
                avg_degree = sum(degrees) / len(degrees) if degrees else 0
                
                # Clustering and connectivity
                clustering = nx.average_clustering(self.graph.to_undirected())
                components = nx.number_weakly_connected_components(self.graph)
                
                # Path metrics - handle disconnected graphs
                if nx.is_weakly_connected(self.graph):
                    try:
                        avg_path = nx.average_shortest_path_length(self.graph)
                        diameter = nx.diameter(self.graph)
                    except nx.NetworkXError:
                        avg_path = float('inf')
                        diameter = None
                else:
                    avg_path = float('inf')
                    diameter = None
                
                # Use degree centrality as it's faster than PageRank
                centrality = nx.degree_centrality(self.graph)
                
                return GraphStats(
                    node_count=node_count,
                    edge_count=edge_count,
                    density=density,
                    avg_degree=avg_degree,
                    clustering_coefficient=clustering,
                    connected_components=components,
                    avg_path_length=avg_path,
                    diameter=diameter,
                    centrality_scores=centrality
                )
                
        except TimeoutError:
            logger.warning("Graph statistics computation timed out")
            return None
        except Exception as e:
            logger.error(f"Error computing graph statistics: {e}")
            return None

    def find_key_theorems(self, top_k: int = 10) -> List[str]:
        """Find the most important theorems using centrality metrics"""
        if not HAVE_NETWORKX or self.nx is None:
            logger.error("networkx required for finding key theorems")
            return []

        try:
            with timeout(20):  # 20 second timeout
                nx = self.nx
                # Use simpler centrality measures for better performance
                scores = {}
                
                # Degree centrality
                degree_cent = nx.degree_centrality(self.graph)
                scores.update(degree_cent)
                
                # Try adding closeness centrality for more insight
                try:
                    closeness = nx.closeness_centrality(self.graph)
                    for node, score in closeness.items():
                        scores[node] = scores.get(node, 0) + score
                except:
                    logger.debug("Skipping closeness centrality")
                
                # Sort and return top theorems
                return sorted(
                    scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:top_k]
                
        except TimeoutError:
            logger.warning("Key theorem computation timed out")
            return []
        except Exception as e:
            logger.error(f"Error finding key theorems: {e}")
            return []

    def detect_communities(
        self,
        resolution: float = 1.0
    ) -> Dict[str, int]:
        """
        Detect theorem communities using available methods
        
        Args:
            resolution: Resolution parameter for community detection
            
        Returns:
            Dict mapping theorem IDs to community IDs
        """
        if not HAVE_NETWORKX or self.nx is None:
            logger.error("networkx required for community detection")
            return {}

        try:
            with timeout(30):  # 30 second timeout
                nx = self.nx
                # Convert to undirected graph for community detection
                undirected = self.graph.to_undirected()
                
                if HAVE_COMMUNITY:
                    # Attempt to use python-louvain if available
                    try:
                        import community.community_louvain as community_louvain
                        return community_louvain.best_partition(
                            undirected,
                            resolution=resolution
                        )
                    except ImportError:
                        logger.warning("Failed to import community_louvain")
                
                # Fall back to connected components
                communities = {}
                for i, component in enumerate(nx.connected_components(undirected)):
                    for node in component:
                        communities[node] = i
                return communities
                
        except TimeoutError:
            logger.warning("Community detection timed out")
            return {}
        except Exception as e:
            logger.error(f"Error detecting communities: {e}")
            return {}

    def find_theorem_clusters(
        self,
        min_size: int = 3,
        similarity_threshold: float = 0.5
    ) -> List[Set[str]]:
        """Find clusters of related theorems"""
        if not HAVE_NETWORKX or self.nx is None:
            logger.error("networkx required for theorem clustering")
            return []

        try:
            with timeout(30):
                nx = self.nx
                # Create similarity graph
                similarity_graph = nx.Graph()
                
                # Add edges based on similarity metrics
                for node1 in self.graph.nodes():
                    for node2 in self.graph.nodes():
                        if node1 >= node2:
                            continue
                            
                        similarity = self._compute_theorem_similarity(node1, node2)
                        if similarity >= similarity_threshold:
                            similarity_graph.add_edge(node1, node2, weight=similarity)
                
                # Find connected components as clusters
                clusters = [
                    component
                    for component in nx.connected_components(similarity_graph)
                    if len(component) >= min_size
                ]
                
                return sorted(clusters, key=len, reverse=True)
                
        except TimeoutError:
            logger.warning("Clustering timed out")
            return []
        except Exception as e:
            logger.error(f"Error finding theorem clusters: {e}")
            return []

    def _compute_theorem_similarity(self, theorem1: str, theorem2: str) -> float:
        """Compute similarity between two theorems"""
        similarity = 0.0
        
        # Get node attributes
        attrs1 = self.graph.nodes[theorem1]
        attrs2 = self.graph.nodes[theorem2]
        
        # Compare proof techniques
        techniques1 = set(attrs1.get('proof_techniques', []))
        techniques2 = set(attrs2.get('proof_techniques', []))
        if techniques1 and techniques2:
            technique_sim = len(techniques1 & techniques2) / len(techniques1 | techniques2)
            similarity += technique_sim * 0.4
        
        # Compare prerequisites
        prereqs1 = set(self.graph.predecessors(theorem1))
        prereqs2 = set(self.graph.predecessors(theorem2))
        if prereqs1 and prereqs2:
            prereq_sim = len(prereqs1 & prereqs2) / len(prereqs1 | prereqs2)
            similarity += prereq_sim * 0.3
        
        # Check for shared concepts
        concepts1 = set(attrs1.get('concepts', []))
        concepts2 = set(attrs2.get('concepts', []))
        if concepts1 and concepts2:
            concept_sim = len(concepts1 & concepts2) / len(concepts1 | concepts2)
            similarity += concept_sim * 0.3
            
        return similarity