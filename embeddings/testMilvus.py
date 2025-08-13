from milvus_provider.milvus_provider import MilvusManager
from embedding_provider.embedding_provider import EmbeddingService
from pymilvus import Collection, connections
import json
from pprint import pprint

def display_complete_metadata(collection, limit=5):
    """Display ALL metadata fields for each chunk"""
    print(f"\n" + "="*80)
    print("COMPLETE METADATA PER CHUNK")
    print("="*80)
    
    # Get all field names from the collection schema
    schema = collection.schema
    all_fields = [field.name for field in schema.fields if field.name != "vector"]
    
    print(f"Available fields in collection: {all_fields}")
    print(f"\nFetching {limit} chunks with ALL metadata...\n")
    
    # Query with all fields except vector (too large to display)
    results = collection.query(
        expr="",  # Get all
        output_fields=all_fields,
        limit=limit
    )
    
    for i, record in enumerate(results, 1):
        print(f"\n{'='*60}")
        print(f"CHUNK {i}")
        print(f"{'='*60}")
        
        # Display each field systematically
        for field in all_fields:
            value = record.get(field)
            
            if field == "text":
                # Show text content with length info
                text_preview = value[:300] if value else "No text"
                print(f"📄 {field.upper()}:")
                print(f"   Length: {len(value) if value else 0} characters")
                print(f"   Preview: {text_preview}...")
                if value and len(value) > 300:
                    print(f"   [... truncated, showing first 300 chars]")
                print()
                
            elif field in ["themes", "keywords", "entities", "top_related", "bottom_related"]:
                # JSON fields - parse and display nicely
                print(f"🏷️  {field.upper()}:")
                try:
                    if isinstance(value, str):
                        parsed = json.loads(value)
                    else:
                        parsed = value
                    
                    if isinstance(parsed, list):
                        if parsed:
                            for item in parsed:
                                print(f"   • {item}")
                        else:
                            print("   (empty list)")
                    elif isinstance(parsed, dict):
                        if parsed:
                            for key, val in parsed.items():
                                print(f"   {key}: {val}")
                        else:
                            print("   (empty dict)")
                    else:
                        print(f"   {parsed}")
                except Exception as e:
                    print(f"   Raw value: {value}")
                    print(f"   (JSON parse error: {e})")
                print()
                
            elif field in ["url", "title", "subtitle"]:
                # Important text fields
                print(f"🔗 {field.upper()}: {value}")
                print()
                
            elif field in ["publication_date", "update_date", "crawl_timestamp"]:
                # Date fields
                print(f"📅 {field.upper()}: {value}")
                print()
                
            elif field in ["file_size", "page_number"]:
                # Numeric fields
                print(f"🔢 {field.upper()}: {value:,}" if value else f"🔢 {field.upper()}: None")
                print()
                
            else:
                # Other fields
                print(f"ℹ️  {field.upper()}: {value}")
                print()


def display_metadata_by_semantic_search(embedding_service, query, k=3):
    """Show complete metadata for semantically relevant chunks"""
    print(f"\n" + "="*80)
    print(f"SEMANTIC SEARCH RESULTS WITH FULL METADATA")
    print(f"Query: '{query}'")
    print("="*80)
    
    try:
        results = embedding_service.vector_store.similarity_search(
            query=query,
            k=k
        )
        
        for i, doc in enumerate(results, 1):
            print(f"\n{'='*60}")
            print(f"SEARCH RESULT {i}")
            print(f"{'='*60}")
            
            # Show relevance score if available
            print(f"📄 TEXT CONTENT:")
            print(f"   Length: {len(doc.page_content)} characters")
            print(f"   Content: {doc.page_content[:400]}...")
            if len(doc.page_content) > 400:
                print(f"   [... truncated]")
            print()
            
            print(f"📋 COMPLETE METADATA:")
            # Sort metadata keys for consistent display
            for key in sorted(doc.metadata.keys()):
                value = doc.metadata[key]
                
                if key in ["themes", "keywords", "entities", "top_related", "bottom_related"]:
                    print(f"   🏷️  {key}:")
                    try:
                        if isinstance(value, str):
                            parsed = json.loads(value)
                        else:
                            parsed = value
                        
                        if isinstance(parsed, list) and parsed:
                            for item in parsed:
                                print(f"      • {item}")
                        elif isinstance(parsed, dict) and parsed:
                            for k, v in parsed.items():
                                print(f"      {k}: {v}")
                        else:
                            print(f"      {parsed}")
                    except:
                        print(f"      {value}")
                        
                elif key == "file_size":
                    print(f"   📊 {key}: {value:,} bytes" if value else f"   📊 {key}: None")
                    
                elif key in ["publication_date", "update_date", "crawl_timestamp"]:
                    print(f"   📅 {key}: {value}")
                    
                else:
                    # Truncate very long values
                    display_value = str(value)
                    if len(display_value) > 100:
                        display_value = display_value[:100] + "..."
                    print(f"   📝 {key}: {display_value}")
            
            print()
            
    except Exception as e:
        print(f"Error in semantic search: {e}")


def compare_chunks_from_same_document(collection, url_pattern, limit=3):
    """Show how chunks from the same document differ"""
    print(f"\n" + "="*80)
    print(f"CHUNKS FROM SAME DOCUMENT")
    print(f"URL pattern: {url_pattern}")
    print("="*80)
    
    results = collection.query(
        expr=f'url like "%{url_pattern}%"',
        output_fields=["pk", "url", "title", "page_number", "text", "themes"],
        limit=limit
    )
    
    print(f"Found {len(results)} chunks from this document:")
    
    for i, record in enumerate(results, 1):
        print(f"\n--- CHUNK {i} ---")
        print(f"Primary Key: {record.get('pk')}")
        print(f"Page Number: {record.get('page_number')}")
        print(f"Title: {record.get('title', 'No title')}")
        
        text = record.get('text', '')
        print(f"Text Length: {len(text)} characters")
        print(f"Text Preview: {text[:200]}...")
        
        themes = record.get('themes', '[]')
        try:
            if isinstance(themes, str):
                themes = json.loads(themes)
            print(f"Themes: {themes}")
        except:
            print(f"Themes (raw): {themes}")


def display_raw_metadata_dump(collection, limit=2):
    """Show completely raw metadata dump for debugging"""
    print(f"\n" + "="*80)
    print("RAW METADATA DUMP (for debugging)")
    print("="*80)
    
    results = collection.query(
        expr="",
        output_fields=["*"],  # Get everything except vector
        limit=limit
    )
    
    for i, record in enumerate(results, 1):
        print(f"\n--- RAW RECORD {i} ---")
        for key, value in record.items():
            if key != "vector":  # Skip vector field
                print(f"{key}: {type(value)} = {value}")
        print()


if __name__ == "__main__":
    # Configuration
    MILVUS_CONFIG = {
        "host": "54.217.166.223",
        "port": "19530",
        "collection_name": "CG_DEMO",
        "connection_args": {"host": "54.217.166.223", "port": "19530"},
    }
    
    print("🔧 Setting up connections...")
    
    # Method 1: Try using MilvusManager (handles connections internally)
    try:
        print("Attempting connection via MilvusManager...")
        manager = MilvusManager(
            connection_args=MILVUS_CONFIG["connection_args"],
            collection_name=MILVUS_CONFIG["collection_name"],
            host=MILVUS_CONFIG["host"],
            port=MILVUS_CONFIG["port"]
        )
        
        if manager.collection:
            print("✅ MilvusManager connection successful")
            collection = manager.collection
            collection.load()
        else:
            raise Exception("MilvusManager collection is None")
            
    except Exception as e:
        print(f"❌ MilvusManager failed: {e}")
        print("Trying direct connection...")
        
        # Method 2: Direct connection approach
        try:
            connections.connect(
                alias="default",
                host=MILVUS_CONFIG["host"],
                port=MILVUS_CONFIG["port"]
            )
            print(f"✅ Direct connection to Milvus at {MILVUS_CONFIG['host']}:{MILVUS_CONFIG['port']}")
            
            collection = Collection(MILVUS_CONFIG["collection_name"])
            collection.load()
            print(f"✅ Loaded collection: {MILVUS_CONFIG['collection_name']}")
            
        except Exception as e2:
            print(f"❌ Direct connection also failed: {e2}")
            print("Cannot proceed without Milvus connection")
            exit(1)
    
    # Check collection status
    try:
        print(f"Total entities: {collection.num_entities:,}")
    except Exception as e:
        print(f"Could not get entity count: {e}")
    
    try:
        # EmbeddingService for semantic search
        print("\nTrying to set up EmbeddingService...")
        try:
            embedding_service = EmbeddingService(
                use_tei=True,
                milvus_config=MILVUS_CONFIG,
                endpoint_name="embedding-endpoint",
                region_name="eu-west-1",
            )
            print("✅ EmbeddingService initialized")
        except Exception as e:
            print(f"⚠️  EmbeddingService failed: {e}")
            print("   Will skip semantic search examples")
            embedding_service = None
        
        # 0. Raw metadata dump first
        display_raw_metadata_dump(collection, limit=2)
        
        # 1. Show complete metadata for first few chunks
        display_complete_metadata(collection, limit=3)
        
        # 2. Show metadata for semantically relevant chunks
        if embedding_service:
            display_metadata_by_semantic_search(
                embedding_service, 
                "Basel Committee customer due diligence", 
                k=2
            )
        else:
            print("\n⚠️  Skipping semantic search - EmbeddingService not available")
        
        # 3. Compare chunks from same document
        compare_chunks_from_same_document(
            collection, 
            "basel-committee", 
            limit=3
        )
        
        # 4. Custom exploration - show unique values for key fields
        print(f"\n" + "="*80)
        print("METADATA FIELD ANALYSIS")
        print("="*80)
        
        # Sample larger set to analyze metadata patterns
        sample_results = collection.query(
            expr="",
            output_fields=["document_type", "lang", "super_category", "crawl_session"],
            limit=100
        )
        
        # Analyze unique values
        fields_to_analyze = ["document_type", "lang", "super_category", "crawl_session"]
        
        for field in fields_to_analyze:
            unique_values = set()
            for record in sample_results:
                value = record.get(field)
                if value:
                    unique_values.add(value)
            
            print(f"\n📊 {field.upper()} - Unique values ({len(unique_values)}):")
            for value in sorted(unique_values):
                print(f"   • {value}")
        
        # 5. Simple query examples
        print(f"\n" + "="*80)
        print("SIMPLE QUERY EXAMPLES")
        print("="*80)
        
        # Show just a few key fields for easier reading
        simple_results = collection.query(
            expr="",
            output_fields=["url", "title", "document_type", "page_number", "file_size"],
            limit=5
        )
        
        for i, record in enumerate(simple_results, 1):
            print(f"\n{i}. {record.get('title', 'No title')}")
            print(f"   URL: {record.get('url', 'No URL')}")
            print(f"   Type: {record.get('document_type', 'Unknown')}")
            print(f"   Page: {record.get('page_number', 'N/A')}")
            print(f"   Size: {record.get('file_size', 0):,} bytes")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()