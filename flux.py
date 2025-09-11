#!/usr/bin/env python3
"""
FLUX Language REPL and Runner
Command-line interface for the FLUX programming language
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.lexer import FLUXLexer
from src.parser import FLUXParser
from src.interpreter import FLUXInterpreter

def run_file(filename: str):
    """Run a FLUX source file"""
    try:
        with open(filename, 'r') as f:
            source = f.read()
        
        # Tokenize
        lexer = FLUXLexer(source)
        tokens = lexer.tokenize()
        
        # Parse
        parser = FLUXParser(tokens)
        ast = parser.parse()
        
        # Interpret
        interpreter = FLUXInterpreter()
        result = interpreter.interpret(ast)
        
        if result is not None:
            print(f"Result: {result}")
            
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except SyntaxError as e:
        print(f"Syntax Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Runtime Error: {e}")
        sys.exit(1)

def repl():
    """Interactive FLUX REPL"""
    print("FLUX Language v0.1.0 - Interactive REPL")
    print("Type 'exit' to quit\n")
    
    interpreter = FLUXInterpreter()
    
    while True:
        try:
            # Read input
            source = input("flux> ")
            
            if source.strip() == 'exit':
                print("Goodbye!")
                break
            
            if not source.strip():
                continue
            
            # Multi-line support
            if source.strip().endswith('{'):
                lines = [source]
                while True:
                    line = input("... ")
                    lines.append(line)
                    if line.strip() == '}':
                        break
                source = '\n'.join(lines)
            
            # Tokenize
            lexer = FLUXLexer(source)
            tokens = lexer.tokenize()
            
            # Parse
            parser = FLUXParser(tokens)
            ast = parser.parse()
            
            # Interpret
            result = interpreter.interpret(ast)
            
            if result is not None:
                print(result)
                
        except KeyboardInterrupt:
            print("\nKeyboard interrupt")
            continue
        except EOFError:
            print("\nGoodbye!")
            break
        except SyntaxError as e:
            print(f"Syntax Error: {e}")
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        # Run file
        run_file(sys.argv[1])
    else:
        # Start REPL
        repl()

if __name__ == "__main__":
    main()