// FLUX AI and Vector Operations Example
// Demonstrates semantic vectors and AI-native operations

function semantic_search_demo() {
    // Create text embeddings (mock implementation)
    let query = "artificial intelligence"
    let query_vec = embed(query)
    
    // Create a knowledge base of embedded documents
    let doc1 = embed("machine learning and neural networks")
    let doc2 = embed("quantum computing and qubits")
    let doc3 = embed("AI and deep learning algorithms")
    
    // Perform semantic similarity search
    print("Query: " + query)
    print("Similarity scores:")
    
    // The ~~ operator performs semantic similarity
    let sim1 = query_vec ~~ doc1
    let sim2 = query_vec ~~ doc2
    let sim3 = query_vec ~~ doc3
    
    print("Doc1 (ML & Neural Networks): " + str(sim1))
    print("Doc2 (Quantum Computing): " + str(sim2))
    print("Doc3 (AI & Deep Learning): " + str(sim3))
}

function vector_operations() {
    // Create vectors
    let v1 = vector(1.0, 2.0, 3.0)
    let v2 = vector(4.0, 5.0, 6.0)
    
    // Vector operations
    print("\nVector Operations:")
    print("v1: [1, 2, 3]")
    print("v2: [4, 5, 6]")
    
    // Dot product using @ operator
    let dot_product = v1 @ v2
    print("Dot product (v1 @ v2): " + str(dot_product))
}

// Run demonstrations
semantic_search_demo()
vector_operations()