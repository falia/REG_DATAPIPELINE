#!/usr/bin/env python3
"""
Test script for EmbeddingService with MilvusManager
Run this to test both local and remote embedding functionality using LangChain methods only
"""

# !/usr/bin/env python3
"""
Test script for EmbeddingService with MilvusManager
"""
import sys
import os


# Ensure we can import from any directory
def find_project_root():
    """Find the project root directory containing embedding_provider"""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up directories until we find embedding_provider
    while current_dir != os.path.dirname(current_dir):  # Not at filesystem root
        if os.path.exists(os.path.join(current_dir, "embedding_provider")):
            return current_dir
        current_dir = os.path.dirname(current_dir)

    raise ImportError("Could not find project root with embedding_provider!")


# Add project root to Python path
project_root = find_project_root()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from embeddings.embedding_provider import EmbeddingService




def test_remote_sagemaker_service():
    """Test Remote SageMaker Service using LangChain methods"""
    print("1. Testing Remote SageMaker Service (LangChain)")
    print("-" * 40)

    milvus_config = {
        "host": "localhost",
        "port": "19530",
        "collection_name": "cssf_test_3",
        "connection_args": {"host": "localhost", "port": "19530"}
    }

    remote_service = EmbeddingService(
        use_remote=True,
        endpoint_name='embedding-endpoint',
        milvus_config=milvus_config,
        region_name='eu-west-1',
    )

    try:
        # Single text addition - LangChain handles embedding
        result = remote_service.add_text_to_store(
            "Your text to embed here",
            metadata={"source": "test_doc", "region": "eu"}
        )
        print(f"Remote text saved: {result['saved_to_milvus']}")
        print(f"Milvus IDs: {result.get('milvus_ids', [])}")
        print(f"Text count: {result['count']}")

        # Batch text addition - LangChain handles embeddings
        texts = ["First document text", "Second document text", "Third document text"]
        metadatas = [
            {"source": "doc1", "category": "research", "region": "eu"},
            {"source": "doc2", "category": "analysis", "region": "eu"},
            {"source": "doc3", "category": "summary", "region": "eu"}
        ]

        batch_result = remote_service.add_texts_to_store(texts, metadatas)
        print(f"Batch saved {batch_result['count']} texts")
        print(f"Batch Milvus IDs: {batch_result.get('milvus_ids', [])}")

        # Search - LangChain handles query embedding
        similar = remote_service.search_similar_texts("Your search query", top_k=3, with_scores=True)
        print(f"Found {len(similar)} similar texts")
        for i, item in enumerate(similar):
            print(f"  {i + 1}. {item.get('content', 'N/A')[:50]}... (score: {item.get('score', 'N/A')})")

        return True

    except Exception as e:
        print(f"Remote service error: {e}")
        return False


def test_local_embedding_service():
    """Test Local Embedding Service using LangChain methods"""
    print("2. Testing Local Embedding Service (LangChain)")
    print("-" * 40)

    local_service = EmbeddingService(
        use_remote=False,
        model_name="BAAI/bge-large-en-v1.5",
        milvus_config={
            "host": "52.31.135.91",
            "port": "19530",
            "collection_name": "local_embeddings",
            "connection_args": {"host": "52.31.135.91", "port": "19530"}
        },
    )

    try:
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand human language."
        ]

        metadatas = [
            {"source": "ml_doc", "category": "AI", "priority": "high"},
            {"source": "dl_doc", "category": "AI", "priority": "medium"},
            {"source": "nlp_doc", "category": "AI", "priority": "high"}
        ]

        # Use LangChain method - no double embedding generation
        result = local_service.add_texts_to_store(documents, metadatas)
        print(f"Local texts saved - Count: {result['count']}")
        print(f"Saved to Milvus: {result['saved_to_milvus']}")
        print(f"Milvus IDs: {result.get('milvus_ids', [])}")

        # Search using LangChain - single embedding generation
        query = "What is artificial intelligence?"
        similar_docs = local_service.search_similar_texts(query, top_k=2, with_scores=True)
        print(f"\nQuery: '{query}'")
        print("Similar documents:")
        for i, doc in enumerate(similar_docs):
            content = doc.get('content', 'N/A')
            score = doc.get('score', 'N/A')
            metadata = doc.get('metadata', {})
            print(f"  {i + 1}. {content} (score: {score})")
            print(f"     Source: {metadata.get('source', 'N/A')}")

        return True

    except Exception as e:
        print(f"Local service error: {e}")
        return False


def test_late_milvus_setup():
    """Test Late Milvus Setup using LangChain methods"""
    print("3. Testing Late Milvus Setup (LangChain)")
    print("-" * 40)

    try:
        service = EmbeddingService(use_remote=False, model_name="BAAI/bge-large-en-v1.5")

        service.setup_milvus(
            host="52.31.135.91",
            port="19530",
            connection_args={"host": "52.31.135.91", "port": "19530"},
            collection_name="late_setup_embeddings"
        )

        # Use LangChain method
        result = service.add_text_to_store(
            "Test document for later setup",
            metadata={"test": "late_setup", "timestamp": "2025-06-15"}
        )
        print(f"Later setup result: {result['saved_to_milvus']}")
        print(f"Text count: {result['count']}")

        # Search using LangChain
        search_results = service.search_similar_texts("test document", top_k=1)
        print(f"Search found {len(search_results)} results")

        return True

    except Exception as e:
        print(f"Late setup error: {e}")
        return False


def test_singleton_connection_reuse():
    """Test Singleton Connection Reuse using LangChain methods"""
    print("4. Testing Singleton Connection Reuse (LangChain)")
    print("-" * 40)

    try:
        service_a = EmbeddingService(
            use_remote=False,
            model_name="BAAI/bge-large-en-v1.5",
            milvus_config={
                "host": "52.31.135.91",
                "port": "19530",
                "collection_name": "collection_a"
            }
        )

        service_b = EmbeddingService(
            use_remote=False,
            model_name="BAAI/bge-large-en-v1.5",
            milvus_config={
                "host": "52.31.135.91",
                "port": "19530",
                "collection_name": "collection_b"
            }
        )

        print(f"MilvusManager instances are same: {service_a.milvus is service_b.milvus}")

        # Use LangChain methods
        service_a.add_text_to_store("Document in collection A", {"collection": "A"})
        service_b.add_text_to_store("Document in collection B", {"collection": "B"})

        print("Successfully added documents to different collections using same connection")

        return True

    except Exception as e:
        print(f"Singleton test error: {e}")
        return False


def test_error_handling():
    """Test Error Handling Scenarios using LangChain methods"""
    print("5. Testing Error Handling (LangChain)")
    print("-" * 40)

    try:
        service = EmbeddingService(use_remote=False, model_name="BAAI/bge-large-en-v1.5")

        # Test search without Milvus configured
        try:
            service.search_similar_texts("test query")
            print("ERROR: Should have failed - no Milvus configured")
            return False
        except Exception as expected_error:
            print(f"✓ Expected error caught: {expected_error}")

        # Setup Milvus
        service.setup_milvus(
            host="52.31.135.91",
            port="19530",
            collection_name="error_test"
        )

        # Test adding text without collection created
        try:
            service.add_text_to_store("test text")
            print("Collection was created automatically or error was handled")
        except Exception as e:
            print(f"Error adding text: {e}")

        # After proper setup, this should work
        result = service.add_text_to_store("Now this should work", {"test": "error_handling"})
        print(f"✓ After proper setup, text saved: {result['saved_to_milvus']}")

        return True

    except Exception as e:
        print(f"Error handling test failed: {e}")
        return False


def test_provider_switching():
    """Test Provider Switching using LangChain methods"""
    print("6. Testing Provider Switching (LangChain)")
    print("-" * 40)

    try:
        service = EmbeddingService(use_remote=False, model_name="BAAI/bge-large-en-v1.5")

        service.setup_milvus(
            host="52.31.135.91",
            port="19530",
            collection_name="provider_switch_test"
        )

        # Test direct embedding creation (not for storage)
        local_embedding = service.create_embedding("Test with local provider")
        print(f"Local embedding dimension: {len(local_embedding)}")

        # Add text using local provider via LangChain
        result1 = service.add_text_to_store("Text with local provider", {"provider": "local"})
        print(f"Local provider text saved: {result1['saved_to_milvus']}")

        # Note: This will fail if SageMaker endpoint isn't available
        try:
            service.switch_provider(use_remote=True, endpoint_name='embedding-endpoint', region_name='eu-west-1')
            remote_embedding = service.create_embedding("Test with remote provider")
            print(f"Remote embedding dimension: {len(remote_embedding)}")

            # Add text using remote provider via LangChain
            result2 = service.add_text_to_store("Text with remote provider", {"provider": "remote"})
            print(f"Remote provider text saved: {result2['saved_to_milvus']}")
            print("✓ Provider switching successful")
        except Exception as e:
            print(f"Remote provider switch failed (expected if no endpoint): {e}")

        # Switch back to local
        service.switch_provider(use_remote=False, model_name="BAAI/bge-large-en-v1.5")
        local_embedding2 = service.create_embedding("Back to local provider")
        print(f"Back to local embedding dimension: {len(local_embedding2)}")

        return True

    except Exception as e:
        print(f"Provider switching test error: {e}")
        return False


def test_embedding_only_operations():
    """Test operations that only need embeddings (no storage)"""
    print("7. Testing Embedding-Only Operations")
    print("-" * 40)

    try:
        # Test both providers for embedding generation only
        local_service = EmbeddingService(use_remote=False, model_name="BAAI/bge-large-en-v1.5")

        # Generate embedding without storage
        test_text = "This is a test document for embedding generation only"
        local_embedding = local_service.create_embedding(test_text)
        print(f"Local embedding dimension: {len(local_embedding)}")
        print(f"Local embedding type: {type(local_embedding)}")

        # Test remote if available
        try:
            remote_service = EmbeddingService(
                use_remote=True,
                endpoint_name='embedding-endpoint',
                region_name='eu-west-1'
            )
            remote_embedding = remote_service.create_embedding(test_text)
            print(f"Remote embedding dimension: {len(remote_embedding)}")
            print(f"Remote embedding type: {type(remote_embedding)}")

            # Compare embeddings (they should be different due to different models)
            print(f"Embeddings are identical: {local_embedding == remote_embedding}")

        except Exception as e:
            print(f"Remote embedding test failed (expected if no endpoint): {e}")

        return True

    except Exception as e:
        print(f"Embedding-only operations test error: {e}")
        return False


def cleanup():
    """Cleanup connections"""
    try:
        from embeddings.milvus_provider.milvus_provider import MilvusManager
        MilvusManager.disconnect()
        print("✓ Disconnected from Milvus")
    except Exception as e:
        print(f"Cleanup error: {e}")


def run_all_tests():
    """Run all tests and provide summary"""
    print("=== Testing EmbeddingService with LangChain Methods ===\n")

    tests = [
        test_remote_sagemaker_service,
        #test_local_embedding_service,
        #test_late_milvus_setup,
        #test_singleton_connection_reuse,
        #test_error_handling,
        #test_provider_switching,
        #test_embedding_only_operations
    ]

    results = []

    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"Test {test_func.__name__} crashed: {e}")
            results.append((test_func.__name__, False))

        print("\n" + "=" * 50 + "\n")

    print("=== TEST SUMMARY ===")
    passed = 0
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nTotal: {passed}/{len(results)} tests passed")

    cleanup()
    return passed == len(results)


def run_minimal_test():
    """Run minimal test for quick verification"""
    print("=== Running Minimal Test ===\n")

    try:
        # Only test remote SageMaker service
        result = test_remote_sagemaker_service()
        if result:
            print("✓ Minimal test PASSED")
        else:
            print("✗ Minimal test FAILED")
        cleanup()
        return result
    except Exception as e:
        print(f"Minimal test crashed: {e}")
        cleanup()
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--minimal":
        success = run_minimal_test()
    else:
        success = run_all_tests()

    exit(0 if success else 1)