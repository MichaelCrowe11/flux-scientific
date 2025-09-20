import * as vscode from 'vscode';
import * as path from 'path';
import { exec, spawn } from 'child_process';
import * as fs from 'fs';

let outputChannel: vscode.OutputChannel;

export function activate(context: vscode.ExtensionContext) {
    console.log('FLUX Scientific Computing Language extension is now active!');

    outputChannel = vscode.window.createOutputChannel('FLUX');

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('flux.compile', compileFluxFile),
        vscode.commands.registerCommand('flux.compileAndRun', compileAndRunFluxFile),
        vscode.commands.registerCommand('flux.validate', validateFluxFile),
        vscode.commands.registerCommand('flux.showAST', showASTFluxFile),
        vscode.commands.registerCommand('flux.benchmark', runBenchmarks)
    );

    // Register document save handler for validation
    context.subscriptions.push(
        vscode.workspace.onDidSaveTextDocument(onDocumentSave)
    );

    // Register completion provider
    context.subscriptions.push(
        vscode.languages.registerCompletionItemProvider(
            'flux',
            new FluxCompletionProvider(),
            '.',
            ' '
        )
    );

    // Register hover provider
    context.subscriptions.push(
        vscode.languages.registerHoverProvider(
            'flux',
            new FluxHoverProvider()
        )
    );

    // Register diagnostics provider
    context.subscriptions.push(
        vscode.languages.registerDocumentSemanticTokensProvider(
            'flux',
            new FluxSemanticTokensProvider(),
            legend
        )
    );
}

const legend = new vscode.SemanticTokensLegend([
    'keyword', 'string', 'number', 'operator', 'variable', 'function', 'comment'
]);

class FluxSemanticTokensProvider implements vscode.DocumentSemanticTokensProvider {
    provideDocumentSemanticTokens(document: vscode.TextDocument): vscode.ProviderResult<vscode.SemanticTokens> {
        const builder = new vscode.SemanticTokensBuilder(legend);

        for (let lineIndex = 0; lineIndex < document.lineCount; lineIndex++) {
            const line = document.lineAt(lineIndex);
            const text = line.text;

            // Simple tokenization - in a real implementation, you'd use a proper lexer
            const tokens = text.split(/\s+/);
            let charIndex = 0;

            for (const token of tokens) {
                const tokenIndex = text.indexOf(token, charIndex);
                if (tokenIndex !== -1) {
                    const tokenType = getTokenType(token);
                    if (tokenType !== null) {
                        builder.push(lineIndex, tokenIndex, token.length, tokenType, 0);
                    }
                    charIndex = tokenIndex + token.length;
                }
            }
        }

        return builder.build();
    }
}

function getTokenType(token: string): number | null {
    const keywords = ['domain', 'equation', 'boundary', 'solver', 'field', 'function', 'let', 'const'];
    const operators = ['=', '+', '-', '*', '/', '^', 'dt', 'dx', 'dy', 'laplacian'];

    if (keywords.includes(token)) return 0; // keyword
    if (token.startsWith('"') || token.startsWith("'")) return 1; // string
    if (/^\d+(\.\d+)?$/.test(token)) return 2; // number
    if (operators.includes(token)) return 3; // operator
    if (token.endsWith('(')) return 5; // function
    if (token.startsWith('//')) return 6; // comment

    return 4; // variable
}

class FluxCompletionProvider implements vscode.CompletionItemProvider {
    provideCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position
    ): vscode.ProviderResult<vscode.CompletionItem[]> {

        const items: vscode.CompletionItem[] = [];

        // Keywords
        const keywords = [
            'domain', 'equation', 'boundary', 'solver', 'field', 'function',
            'heat', 'wave', 'poisson', 'navier_stokes',
            'dirichlet', 'neumann', 'robin',
            'explicit', 'implicit', 'crank_nicolson'
        ];

        for (const keyword of keywords) {
            const item = new vscode.CompletionItem(keyword, vscode.CompletionItemKind.Keyword);
            item.detail = `FLUX keyword: ${keyword}`;
            items.push(item);
        }

        // Functions
        const functions = [
            { name: 'laplacian', detail: 'Laplacian operator', snippet: 'laplacian(${1:field})' },
            { name: 'gradient', detail: 'Gradient operator', snippet: 'gradient(${1:field})' },
            { name: 'divergence', detail: 'Divergence operator', snippet: 'divergence(${1:field})' },
            { name: 'dt', detail: 'Time derivative', snippet: 'dt(${1:field})' },
            { name: 'dx', detail: 'Spatial derivative (x)', snippet: 'dx(${1:field})' },
            { name: 'dy', detail: 'Spatial derivative (y)', snippet: 'dy(${1:field})' }
        ];

        for (const func of functions) {
            const item = new vscode.CompletionItem(func.name, vscode.CompletionItemKind.Function);
            item.detail = func.detail;
            item.insertText = new vscode.SnippetString(func.snippet);
            items.push(item);
        }

        return items;
    }
}

class FluxHoverProvider implements vscode.HoverProvider {
    provideHover(
        document: vscode.TextDocument,
        position: vscode.Position
    ): vscode.ProviderResult<vscode.Hover> {

        const range = document.getWordRangeAtPosition(position);
        if (!range) return;

        const word = document.getText(range);
        const hoverText = getHoverText(word);

        if (hoverText) {
            return new vscode.Hover(hoverText);
        }
    }
}

function getHoverText(word: string): string | null {
    const documentation: { [key: string]: string } = {
        'domain': 'Defines the computational domain for the PDE',
        'equation': 'Defines a partial differential equation',
        'boundary': 'Specifies boundary conditions',
        'solver': 'Configures the numerical solver',
        'laplacian': 'Laplacian operator (∇²)',
        'gradient': 'Gradient operator (∇)',
        'dt': 'Time derivative (∂/∂t)',
        'dx': 'Spatial derivative (∂/∂x)',
        'dy': 'Spatial derivative (∂/∂y)',
        'dirichlet': 'Dirichlet boundary condition (fixed value)',
        'neumann': 'Neumann boundary condition (fixed derivative)',
        'explicit': 'Explicit time stepping method',
        'implicit': 'Implicit time stepping method',
        'crank_nicolson': 'Crank-Nicolson method (2nd order implicit)'
    };

    return documentation[word] || null;
}

async function compileFluxFile(uri?: vscode.Uri) {
    const activeEditor = vscode.window.activeTextEditor;
    const fileUri = uri || activeEditor?.document.uri;

    if (!fileUri || !fileUri.fsPath.endsWith('.flux')) {
        vscode.window.showErrorMessage('Please select a .flux file to compile');
        return;
    }

    const config = vscode.workspace.getConfiguration('flux');
    const backend = config.get<string>('compiler.backend', 'python');
    const outputDir = config.get<string>('compiler.outputDirectory', 'output');

    outputChannel.show();
    outputChannel.appendLine(`Compiling ${fileUri.fsPath} to ${backend}...`);

    const pythonPath = config.get<string>('python.path', 'python');
    const fluxCompilerPath = findFluxCompiler(fileUri.fsPath);

    if (!fluxCompilerPath) {
        vscode.window.showErrorMessage('FLUX compiler not found. Please ensure flux_scientific.py is in your project.');
        return;
    }

    const command = `${pythonPath} "${fluxCompilerPath}" "${fileUri.fsPath}" -b ${backend} -o "${outputDir}"`;

    exec(command, { cwd: path.dirname(fileUri.fsPath) }, (error, stdout, stderr) => {
        if (error) {
            outputChannel.appendLine(`Error: ${error.message}`);
            vscode.window.showErrorMessage(`Compilation failed: ${error.message}`);
            return;
        }

        if (stderr) {
            outputChannel.appendLine(`Stderr: ${stderr}`);
        }

        outputChannel.appendLine(stdout);
        vscode.window.showInformationMessage(`Compilation successful! Output in ${outputDir}/`);
    });
}

async function compileAndRunFluxFile(uri?: vscode.Uri) {
    const activeEditor = vscode.window.activeTextEditor;
    const fileUri = uri || activeEditor?.document.uri;

    if (!fileUri || !fileUri.fsPath.endsWith('.flux')) {
        vscode.window.showErrorMessage('Please select a .flux file to compile and run');
        return;
    }

    const config = vscode.workspace.getConfiguration('flux');
    const pythonPath = config.get<string>('python.path', 'python');
    const fluxCompilerPath = findFluxCompiler(fileUri.fsPath);

    if (!fluxCompilerPath) {
        vscode.window.showErrorMessage('FLUX compiler not found.');
        return;
    }

    outputChannel.show();
    outputChannel.appendLine(`Compiling and running ${fileUri.fsPath}...`);

    const command = `${pythonPath} "${fluxCompilerPath}" "${fileUri.fsPath}" --run`;

    exec(command, { cwd: path.dirname(fileUri.fsPath) }, (error, stdout, stderr) => {
        if (error) {
            outputChannel.appendLine(`Error: ${error.message}`);
            vscode.window.showErrorMessage(`Execution failed: ${error.message}`);
            return;
        }

        if (stderr) {
            outputChannel.appendLine(`Stderr: ${stderr}`);
        }

        outputChannel.appendLine(stdout);
        vscode.window.showInformationMessage('Execution completed!');
    });
}

async function validateFluxFile(uri?: vscode.Uri) {
    const activeEditor = vscode.window.activeTextEditor;
    const fileUri = uri || activeEditor?.document.uri;

    if (!fileUri || !fileUri.fsPath.endsWith('.flux')) {
        vscode.window.showErrorMessage('Please select a .flux file to validate');
        return;
    }

    const config = vscode.workspace.getConfiguration('flux');
    const pythonPath = config.get<string>('python.path', 'python');
    const fluxCompilerPath = findFluxCompiler(fileUri.fsPath);

    if (!fluxCompilerPath) {
        vscode.window.showErrorMessage('FLUX compiler not found.');
        return;
    }

    const command = `${pythonPath} "${fluxCompilerPath}" "${fileUri.fsPath}" --validate`;

    exec(command, { cwd: path.dirname(fileUri.fsPath) }, (error, stdout, stderr) => {
        if (error) {
            vscode.window.showErrorMessage(`Validation failed: ${error.message}`);
            return;
        }

        if (stdout.includes('✓ Syntax validation passed')) {
            vscode.window.showInformationMessage('FLUX syntax validation passed!');
        } else {
            vscode.window.showWarningMessage('FLUX syntax validation failed. Check the output for details.');
            outputChannel.show();
            outputChannel.appendLine(stdout);
        }
    });
}

async function showASTFluxFile(uri?: vscode.Uri) {
    const activeEditor = vscode.window.activeTextEditor;
    const fileUri = uri || activeEditor?.document.uri;

    if (!fileUri || !fileUri.fsPath.endsWith('.flux')) {
        vscode.window.showErrorMessage('Please select a .flux file to show AST');
        return;
    }

    const config = vscode.workspace.getConfiguration('flux');
    const pythonPath = config.get<string>('python.path', 'python');
    const fluxCompilerPath = findFluxCompiler(fileUri.fsPath);

    if (!fluxCompilerPath) {
        vscode.window.showErrorMessage('FLUX compiler not found.');
        return;
    }

    const command = `${pythonPath} "${fluxCompilerPath}" "${fileUri.fsPath}" --ast`;

    exec(command, { cwd: path.dirname(fileUri.fsPath) }, (error, stdout, stderr) => {
        if (error) {
            vscode.window.showErrorMessage(`AST generation failed: ${error.message}`);
            return;
        }

        // Create a new document with AST output
        vscode.workspace.openTextDocument({
            content: stdout,
            language: 'plaintext'
        }).then(doc => {
            vscode.window.showTextDocument(doc);
        });
    });
}

async function runBenchmarks() {
    const config = vscode.workspace.getConfiguration('flux');
    const pythonPath = config.get<string>('python.path', 'python');

    // Find flux compiler in workspace
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders) {
        vscode.window.showErrorMessage('No workspace folder found');
        return;
    }

    const fluxCompilerPath = path.join(workspaceFolders[0].uri.fsPath, 'flux_scientific.py');
    if (!fs.existsSync(fluxCompilerPath)) {
        vscode.window.showErrorMessage('FLUX compiler not found in workspace root');
        return;
    }

    outputChannel.show();
    outputChannel.appendLine('Running FLUX benchmarks...');

    const command = `${pythonPath} "${fluxCompilerPath}" --benchmark`;

    exec(command, { cwd: workspaceFolders[0].uri.fsPath }, (error, stdout, stderr) => {
        if (error) {
            outputChannel.appendLine(`Error: ${error.message}`);
            vscode.window.showErrorMessage(`Benchmark failed: ${error.message}`);
            return;
        }

        if (stderr) {
            outputChannel.appendLine(`Stderr: ${stderr}`);
        }

        outputChannel.appendLine(stdout);
        vscode.window.showInformationMessage('Benchmarks completed!');
    });
}

function findFluxCompiler(startPath: string): string | null {
    let currentDir = path.dirname(startPath);

    // Look for flux_scientific.py in current dir and parent dirs
    while (currentDir !== path.dirname(currentDir)) {
        const compilerPath = path.join(currentDir, 'flux_scientific.py');
        if (fs.existsSync(compilerPath)) {
            return compilerPath;
        }
        currentDir = path.dirname(currentDir);
    }

    return null;
}

function onDocumentSave(document: vscode.TextDocument) {
    if (document.languageId === 'flux') {
        const config = vscode.workspace.getConfiguration('flux');
        const validateOnSave = config.get<boolean>('validation.onSave', true);

        if (validateOnSave) {
            validateFluxFile(document.uri);
        }
    }
}

export function deactivate() {
    if (outputChannel) {
        outputChannel.dispose();
    }
}