// Fibonacci sequence in FLUX
// Demonstrates functions, recursion, and loops

// Recursive implementation
function fib_recursive(n: int) -> int {
    if n <= 1 {
        return n
    }
    return fib_recursive(n - 1) + fib_recursive(n - 2)
}

// Iterative implementation
function fib_iterative(n: int) -> int {
    if n <= 1 {
        return n
    }
    
    let prev = 0
    let curr = 1
    let i = 2
    
    while i <= n {
        let next = prev + curr
        prev = curr
        curr = next
        i = i + 1
    }
    
    return curr
}

// Generate Fibonacci sequence
function generate_sequence(count: int) {
    print("First " + str(count) + " Fibonacci numbers:")
    
    let i = 0
    while i < count {
        let fib_num = fib_iterative(i)
        print("F(" + str(i) + ") = " + str(fib_num))
        i = i + 1
    }
}

// Main execution
function main() {
    // Test recursive version
    let n = 10
    print("Fibonacci (recursive) of " + str(n) + ": " + str(fib_recursive(n)))
    
    // Test iterative version
    print("Fibonacci (iterative) of " + str(n) + ": " + str(fib_iterative(n)))
    
    // Generate sequence
    print("")
    generate_sequence(15)
}

main()