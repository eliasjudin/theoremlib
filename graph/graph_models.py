"""
Graph Models Module

Defines SQLAlchemy ORM models for storing theorem-proof pairs and their metadata,
with support for vector embeddings using pgvector.
"""

from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Float, JSON, Enum as SQLEnum
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from datetime import datetime
from enum import Enum
import logging
from pydantic import BaseModel, Field, validator
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional dependency flags
HAVE_NETWORKX = False
HAVE_COMMUNITY = False

# Try importing optional dependencies
try:
    import networkx
    HAVE_NETWORKX = True
    logger.info("NetworkX available for graph operations")
except ImportError:
    logger.warning("networkx not available, graph operations will be limited")

try:
    from community import community_louvain
    HAVE_COMMUNITY = True
    logger.info("Community detection available")
except ImportError:
    logger.warning("python-louvain not available, community detection will be limited")

Base = declarative_base()

class EdgeType(str, Enum):
    """Valid types of edges in the theorem graph"""
    LOGICAL_DEPENDENCY = "logical_dependency"  # Theorem A requires Theorem B
    PROOF_TECHNIQUE = "proof_technique"     # Theorems share proof techniques
    EQUIVALENCE = "equivalence"             # Theorems are equivalent/isomorphic

class GraphNodeMetadata(BaseModel):
    """Metadata for graph nodes"""
    impact_score: float = Field(0.0, ge=0.0, le=1.0)
    centrality: Optional[float] = None
    community_id: Optional[int] = None
    tags: List[str] = Field(default_factory=list)

    class Config:
        frozen = True

class GraphEdgeMetadata(BaseModel):
    """Metadata for graph edges"""
    confidence: float = Field(..., ge=0.0, le=1.0)
    description: Optional[str] = None
    references: List[str] = Field(default_factory=list)
    bidirectional: bool = False

    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Confidence score must be between 0 and 1')
        return v

    class Config:
        frozen = True

class GraphNode(Base):
    """Graph node representing a theorem"""
    __tablename__ = "graph_nodes"

    id = Column(Integer, primary_key=True)
    theorem_id = Column(String, ForeignKey("theorems.id"), unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON, default=dict)

    # Relationships
    theorem = relationship("Theorem", backref=backref("graph_node", uselist=False))
    outgoing_edges = relationship(
        "GraphEdge",
        primaryjoin="GraphNode.id==GraphEdge.source_node_id",
        backref="source_node"
    )
    incoming_edges = relationship(
        "GraphEdge",
        primaryjoin="GraphNode.id==GraphEdge.target_node_id",
        backref="target_node"
    )

class GraphEdge(Base):
    """Graph edge representing relationships between theorems"""
    __tablename__ = "graph_edges"

    id = Column(Integer, primary_key=True)
    source_node_id = Column(Integer, ForeignKey("graph_nodes.id"))
    target_node_id = Column(Integer, ForeignKey("graph_nodes.id"))
    edge_type = Column(SQLEnum(EdgeType))
    weight = Column(Float, default=1.0)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class TheoremGraph:
    """Manager class for theorem dependency graph operations"""
    
    def __init__(self):
        if not HAVE_NETWORKX:
            raise ImportError(
                "networkx is required for graph operations. "
                "Install it with: pip install networkx"
            )
        self.graph = networkx.MultiDiGraph()
        self.edge_metadata: Dict[Tuple[str, str, str], GraphEdgeMetadata] = {}

    async def create_edge(
        self,
        session: AsyncSession,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        metadata: GraphEdgeMetadata
    ) -> GraphEdge:
        """
        Create a new edge between theorem nodes
        
        Args:
            session: Database session
            source_id: Source theorem ID
            target_id: Target theorem ID
            edge_type: Type of relationship
            metadata: Edge metadata
            
        Returns:
            Created GraphEdge
        """
        try:
            # Get or create nodes
            source_node = await self._get_or_create_node(session, source_id)
            target_node = await self._get_or_create_node(session, target_id)
            
            # Create edge
            edge = GraphEdge(
                source_node_id=source_node.id,
                target_node_id=target_node.id,
                edge_type=edge_type,
                weight=metadata.confidence,
                metadata=metadata.dict()
            )
            
            session.add(edge)
            
            # Update NetworkX graph
            self.graph.add_edge(
                source_id,
                target_id,
                key=edge_type,
                weight=metadata.confidence
            )
            self.edge_metadata[(source_id, target_id, edge_type)] = metadata
            
            await session.commit()
            await session.refresh(edge)
            
            logger.info(
                f"Created {edge_type} edge from {source_id} to {target_id} "
                f"with confidence {metadata.confidence:.2f}"
            )
            
            return edge

        except Exception as e:
            await session.rollback()
            logger.error(f"Error creating graph edge: {e}")
            raise

    async def _get_or_create_node(
        self,
        session: AsyncSession,
        theorem_id: str
    ) -> GraphNode:
        """Get existing node or create a new one"""
        node = await session.get(GraphNode, {"theorem_id": theorem_id})
        if not node:
            node = GraphNode(theorem_id=theorem_id)
            session.add(node)
        return node

    async def get_dependencies(
        self,
        session: AsyncSession,
        theorem_id: str,
        depth: int = 1,
        edge_types: Optional[Set[EdgeType]] = None
    ) -> networkx.DiGraph:
        """
        Get a subgraph of dependencies up to specified depth
        
        Args:
            session: Database session
            theorem_id: Root theorem ID
            depth: Maximum traversal depth
            edge_types: Types of edges to include
            
        Returns:
            NetworkX DiGraph of dependencies
        """
        if edge_types is None:
            edge_types = set(EdgeType)

        # Create NetworkX graph for traversal
        subgraph = networkx.DiGraph()
        visited = set()

        async def traverse(current_id: str, current_depth: int):
            if current_depth > depth or current_id in visited:
                return
            visited.add(current_id)

            # Get outgoing edges
            edges = await session.execute(f"""
                SELECT ge.* FROM graph_edges ge
                JOIN graph_nodes gn1 ON ge.source_node_id = gn1.id
                JOIN graph_nodes gn2 ON ge.target_node_id = gn2.id
                JOIN theorems t1 ON gn1.theorem_id = t1.id
                JOIN theorems t2 ON gn2.theorem_id = t2.id
                WHERE t1.id = :theorem_id
                AND edge_type = ANY(:edge_types)
            """, {
                "theorem_id": current_id,
                "edge_types": [e.value for e in edge_types]
            })

            for edge in edges:
                target_id = edge.target_node_id
                subgraph.add_edge(
                    current_id,
                    target_id,
                    type=edge.edge_type,
                    weight=edge.weight,
                    metadata=edge.metadata
                )
                await traverse(target_id, current_depth + 1)

        await traverse(theorem_id, 0)
        return subgraph

    async def find_proof_technique_clusters(
        self,
        session: AsyncSession
    ) -> List[Set[str]]:
        """
        Find clusters of theorems sharing similar proof techniques
        
        Returns:
            List of theorem ID sets representing clusters
        """
        technique_edges = await session.execute(f"""
            SELECT DISTINCT ge.* FROM graph_edges ge
            WHERE edge_type = :edge_type
        """, {"edge_type": EdgeType.PROOF_TECHNIQUE.value})
        
        if not HAVE_NETWORKX:
            logger.error("networkx required for cluster detection")
            return []
        
        # Build graph for community detection
        technique_graph = networkx.Graph()
        for edge in technique_edges:
            technique_graph.add_edge(
                edge.source_node_id,
                edge.target_node_id,
                weight=edge.weight
            )

        # Use Louvain method if available, otherwise fall back to connected components
        if HAVE_COMMUNITY:
            partition = community_louvain.best_partition(technique_graph)
            
            # Group theorems by community
            communities: Dict[int, Set[str]] = {}
            for node, community_id in partition.items():
                if community_id not in communities:
                    communities[community_id] = set()
                communities[community_id].add(node)
            
            return list(communities.values())
        else:
            logger.info("Using connected components for clustering (less optimal)")
            return [set(c) for c in networkx.connected_components(technique_graph)]

    async def find_equivalent_theorems(
        self,
        session: AsyncSession,
        theorem_id: str
    ) -> Set[str]:
        """
        Find all theorems equivalent to the given theorem
        
        Args:
            session: Database session
            theorem_id: Theorem to find equivalences for
            
        Returns:
            Set of equivalent theorem IDs
        """
        # Get equivalence edges
        edges = await session.execute(f"""
            SELECT ge.* FROM graph_edges ge
            JOIN graph_nodes gn1 ON ge.source_node_id = gn1.id
            JOIN graph_nodes gn2 ON ge.target_node_id = gn2.id
            WHERE edge_type = :edge_type
            AND (gn1.theorem_id = :theorem_id OR gn2.theorem_id = :theorem_id)
        """, {
            "edge_type": EdgeType.EQUIVALENCE.value,
            "theorem_id": theorem_id
        })

        # Create undirected graph of equivalences
        equiv_graph = networkx.Graph()
        for edge in edges:
            equiv_graph.add_edge(edge.source_node_id, edge.target_node_id)

        # Return connected component containing theorem
        if theorem_id in equiv_graph:
            return set(networkx.node_connected_component(equiv_graph, theorem_id))
        return {theorem_id}

    async def calculate_theorem_impact(
        self,
        session: AsyncSession,
        theorem_id: str
    ) -> float:
        """
        Calculate impact score based on dependencies and confidence
        
        Args:
            session: Database session
            theorem_id: Theorem to calculate impact for
            
        Returns:
            Impact score between 0 and 1
        """
        # Get all theorems that depend on this one
        subgraph = await self.get_dependencies(
            session, theorem_id, depth=None,
            edge_types={EdgeType.LOGICAL_DEPENDENCY}
        )
        
        if not subgraph:
            return 0.0
            
        # Calculate weighted impact
        impact = 0.0
        for edge in subgraph.edges(data=True):
            impact += edge.get('weight', 1.0)
            
        # Normalize by number of edges
        if subgraph.number_of_edges() > 0:
            impact /= subgraph.number_of_edges()
            
        return min(impact, 1.0)