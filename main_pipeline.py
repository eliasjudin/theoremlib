"""
Main Pipeline Orchestration Module

Coordinates the complete theorem processing pipeline from PDF ingestion through
graph node creation, with comprehensive error handling and logging.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import asyncio

from ingest.ingest import PDFIngestor, DocumentMetadata
from ocr.ocr_module import OCRProcessor, OCRResult
from extraction.content_extraction import ContentExtractor, TheoremProofPair
from annotation.metadata_annotation import MetadataAnnotator, TheoremAnnotation, SourceMetadata
from database.database_loader import DatabaseLoader
from graph.graph_models import GraphNode, EdgeType, GraphEdgeMetadata, TheoremGraph
from database.models import TheoremCreate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineCoordinator:
    """Main coordinator for the theorem processing pipeline"""
    
    def __init__(
        self,
        db_url: str,
        storage_path: Path,
        ontology_path: Path,
        ocr_api_key: str
    ):
        """Initialize pipeline components"""
        self.pdf_ingestor = PDFIngestor(storage_path)
        self.ocr_processor = OCRProcessor(ocr_api_key)
        self.content_extractor = ContentExtractor()
        self.metadata_annotator = MetadataAnnotator(ontology_path)
        self.db_loader = DatabaseLoader(db_url)
        self.theorem_graph = TheoremGraph()

    async def process_document(
        self,
        file_path: str,
        metadata: DocumentMetadata
    ) -> List[str]:
        """
        Process a document through the complete pipeline
        
        Args:
            file_path: Path to the PDF file
            metadata: Document metadata
            
        Returns:
            List of created theorem IDs
        """
        try:
            # Step 1: Ingest PDF and validate
            logger.info(f"Starting ingestion of {file_path}")
            pdf_metadata = await self.pdf_ingestor.ingest_pdf(
                file_path,
                {
                    "title": metadata.title,
                    "authors": [metadata.author],
                    "source": metadata.source
                },
                metadata
            )
            stored_path = pdf_metadata.document_metadata.file_hash

            # Step 2: Process OCR if needed or extract LaTeX
            logger.info("Starting text extraction")
            ocr_results = await self.ocr_processor.process_pdf(Path(file_path))
            
            # Step 3: Extract theorem-proof pairs from text
            logger.info("Extracting theorem-proof pairs")
            all_theorems = []
            for result in ocr_results:
                pairs = self.content_extractor.extract_pairs(
                    result.text_content,
                    result.spatial_layout,
                    result.page_number
                )
                all_theorems.extend(pairs)

            # Step 4: Create source metadata for annotation
            source_meta = SourceMetadata(
                source=metadata.source,
                author=metadata.author,
                title=metadata.title,
                page_number=1,  # Will be updated per theorem
                theorem_number=metadata.theorem_number
            )

            # Step 5: Process each theorem
            theorem_ids = []
            async with self.db_loader.get_db() as session:
                for pair in all_theorems:
                    try:
                        # Update page number in source metadata
                        source_meta.page_number = pair.page_number
                        source_meta.theorem_number = pair.theorem_number

                        # Annotate theorem with metadata
                        annotation = await self.metadata_annotator.annotate_theorem(
                            pair.theorem_text,
                            pair.proof_text,
                            source_meta
                        )

                        # Create theorem record
                        theorem_data = TheoremCreate(
                            theorem_type=pair.theorem_type,
                            theorem_number=pair.theorem_number,
                            theorem_title=None,  # Optional
                            theorem_text=pair.theorem_text,
                            proof_text=pair.proof_text,
                            page_number=pair.page_number,
                            source_id=stored_path,
                            proof_techniques=annotation.proof_techniques,
                            difficulty_score=annotation.difficulty_score,
                            concepts=[c.id for c in annotation.key_concepts],
                            spatial_info=pair.spatial_info,
                            formatting=pair.formatting,
                            ontology_mapping=annotation.ontology_mapping
                        )

                        # Store in database
                        theorem_id = await self.db_loader.store_theorem(
                            session,
                            theorem_data
                        )
                        theorem_ids.append(theorem_id)

                        # Create graph node
                        node = GraphNode(
                            theorem_id=theorem_id,
                            metadata={
                                "difficulty": annotation.difficulty_score,
                                "techniques": [t.value for t in annotation.proof_techniques],
                                "concepts": [c.id for c in annotation.key_concepts]
                            }
                        )
                        session.add(node)

                        # Connect to prerequisite nodes
                        for prereq_id in annotation.prerequisites:
                            edge_meta = GraphEdgeMetadata(
                                confidence=0.8,
                                description=f"Prerequisite concept: {prereq_id}"
                            )
                            await self.theorem_graph.create_edge(
                                session,
                                theorem_id,
                                prereq_id,
                                EdgeType.LOGICAL_DEPENDENCY,
                                edge_meta
                            )

                    except Exception as e:
                        logger.error(f"Error processing theorem: {e}")
                        continue

                await session.commit()
                
                # Step 6: Update graph with new nodes
                await self._update_graph_with_new_nodes(session, theorem_ids)

            return theorem_ids

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise

    async def _update_graph_with_new_nodes(
        self,
        session: Any,
        theorem_ids: List[str]
    ) -> None:
        """Update graph with new theorem nodes"""
        try:
            # Find related theorems based on concepts and techniques
            for theorem_id in theorem_ids:
                similar = await self.db_loader.vector_search(
                    session,
                    theorem_id,
                    limit=5,
                    min_similarity=0.7
                )
                
                # Create edges for similar theorems
                for similar_theorem in similar:
                    if similar_theorem.id != theorem_id:
                        edge_meta = GraphEdgeMetadata(
                            confidence=0.7,
                            description="Similar theorem based on content"
                        )
                        await self.theorem_graph.create_edge(
                            session,
                            theorem_id,
                            similar_theorem.id,
                            EdgeType.PROOF_TECHNIQUE,
                            edge_meta
                        )

        except Exception as e:
            logger.error(f"Error updating graph: {e}")
            raise

async def main():
    """Example usage of the pipeline"""
    coordinator = PipelineCoordinator(
        db_url="postgresql+asyncpg://user:pass@localhost/theoremlib",
        storage_path=Path("storage"),
        ontology_path=Path("ontology/math_ontology.json"),
        ocr_api_key="your-api-key"
    )
    
    metadata = DocumentMetadata(
        source="Example Journal",
        title="Sample Theorem Collection",
        author="John Doe",
        page_count=10,
        theorem_number="1"
    )
    
    theorem_ids = await coordinator.process_document("path/to/document.pdf", metadata)
    print(f"Processed {len(theorem_ids)} theorems")

if __name__ == "__main__":
    asyncio.run(main())