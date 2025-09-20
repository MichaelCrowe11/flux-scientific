// FLUX Web App JavaScript

let codeEditor;

// Initialize CodeMirror editor
document.addEventListener('DOMContentLoaded', function() {
    const textArea = document.getElementById('code-editor');
    if (textArea) {
        codeEditor = CodeMirror.fromTextArea(textArea, {
            mode: 'text/plain',
            theme: 'default',
            lineNumbers: true,
            lineWrapping: true,
            indentUnit: 4,
            indentWithTabs: false
        });
    }

    // Setup event listeners
    setupEventListeners();
});

function setupEventListeners() {
    // Range sliders
    const rangeSliders = [
        { id: 'grid-size', label: 'grid-size-label', format: v => `${v}×${v}` },
        { id: 'alpha', label: 'alpha-label', format: v => v },
        { id: 'time-final', label: 'time-final-label', format: v => v }
    ];

    rangeSliders.forEach(slider => {
        const element = document.getElementById(slider.id);
        const label = document.getElementById(slider.label);
        if (element && label) {
            element.addEventListener('input', (e) => {
                label.textContent = slider.format(e.target.value);
            });
        }
    });
}

// Tab switching
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });

    // Remove active from all buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');

    // Mark button as active
    event.target.classList.add('active');
}

// Load example code
async function loadExample() {
    const select = document.getElementById('example-select');
    const example = select.value;

    if (!example) return;

    try {
        const response = await fetch('/api/examples');
        const examples = await response.json();

        if (examples[example]) {
            codeEditor.setValue(examples[example]);
        }
    } catch (error) {
        console.error('Failed to load example:', error);
        showMessage('Failed to load example', 'error');
    }
}

// Validate FLUX code
async function validateCode() {
    const code = codeEditor.getValue();

    if (!code.trim()) {
        showMessage('Please enter some FLUX code', 'error');
        return;
    }

    try {
        showSpinner();
        const response = await fetch('/api/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code })
        });

        const result = await response.json();
        hideSpinner();

        if (result.valid) {
            showMessage(`✓ Syntax is valid! (${result.tokens} tokens, ${result.ast_nodes} AST nodes)`, 'success');
        } else {
            showMessage(`✗ Syntax error: ${result.error}`, 'error');
        }
    } catch (error) {
        hideSpinner();
        showMessage('Validation failed', 'error');
    }
}

// Compile FLUX code
async function compileCode() {
    const code = codeEditor.getValue();
    const backend = document.getElementById('backend-select').value;

    if (!code.trim()) {
        showMessage('Please enter some FLUX code', 'error');
        return;
    }

    try {
        showSpinner();
        const response = await fetch('/api/compile', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code, backend })
        });

        const result = await response.json();
        hideSpinner();

        if (result.success) {
            const outputDiv = document.getElementById('compile-output');
            const codeOutput = document.getElementById('generated-code');

            outputDiv.classList.remove('hidden');
            codeOutput.textContent = result.generated_code;

            showMessage(`✓ Compiled to ${backend} successfully!`, 'success');

            // Scroll to output
            outputDiv.scrollIntoView({ behavior: 'smooth' });
        } else {
            showMessage(`Compilation error: ${result.error}`, 'error');
        }
    } catch (error) {
        hideSpinner();
        showMessage('Compilation failed', 'error');
    }
}

// Solve PDE
async function solvePDE() {
    const params = {
        pde_type: document.getElementById('pde-type').value,
        nx: parseInt(document.getElementById('grid-size').value),
        ny: parseInt(document.getElementById('grid-size').value),
        initial_condition: document.getElementById('initial-condition').value,
        method: document.getElementById('method').value,
        alpha: parseFloat(document.getElementById('alpha').value),
        time_final: parseFloat(document.getElementById('time-final').value),
        dt: 0.01,
        boundary_value: 0.0,
        wave_speed: parseFloat(document.getElementById('alpha').value),
        source_strength: 1.0
    };

    try {
        showSpinner();
        const response = await fetch('/api/solve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });

        const result = await response.json();
        hideSpinner();

        if (result.success) {
            // Display solution
            const outputDiv = document.getElementById('solution-output');
            const plotImg = document.getElementById('solution-plot');
            const statsDiv = document.getElementById('solution-stats');

            outputDiv.classList.remove('hidden');
            plotImg.src = `data:image/png;base64,${result.plot}`;

            // Display statistics
            statsDiv.innerHTML = `
                <h4>Solution Statistics</h4>
                <p><strong>Grid Size:</strong> ${result.grid_size}</p>
                <p><strong>PDE Type:</strong> ${result.pde_type}</p>
                <p><strong>Max Value:</strong> ${result.stats.max_value.toFixed(6)}</p>
                <p><strong>Min Value:</strong> ${result.stats.min_value.toFixed(6)}</p>
                <p><strong>Mean Value:</strong> ${result.stats.mean_value.toFixed(6)}</p>
                ${result.stats.convergence ? `<p><strong>Iterations:</strong> ${result.stats.convergence}</p>` : ''}
            `;

            // Scroll to output
            outputDiv.scrollIntoView({ behavior: 'smooth' });

            showMessage('✓ PDE solved successfully!', 'success');
        } else {
            showMessage(`Solver error: ${result.error}`, 'error');
        }
    } catch (error) {
        hideSpinner();
        showMessage('Failed to solve PDE', 'error');
    }
}

// Utility functions
function showMessage(message, type = 'info') {
    // Create message element
    const messageDiv = document.createElement('div');
    messageDiv.className = type;
    messageDiv.textContent = message;
    messageDiv.style.position = 'fixed';
    messageDiv.style.top = '20px';
    messageDiv.style.right = '20px';
    messageDiv.style.zIndex = '9999';
    messageDiv.style.padding = '15px 20px';
    messageDiv.style.borderRadius = '5px';
    messageDiv.style.boxShadow = '0 4px 6px rgba(0,0,0,0.1)';
    messageDiv.style.animation = 'slideIn 0.3s';

    document.body.appendChild(messageDiv);

    // Remove after 3 seconds
    setTimeout(() => {
        messageDiv.style.animation = 'slideOut 0.3s';
        setTimeout(() => messageDiv.remove(), 300);
    }, 3000);
}

function showSpinner() {
    const spinner = document.createElement('div');
    spinner.className = 'spinner';
    spinner.id = 'global-spinner';
    spinner.style.position = 'fixed';
    spinner.style.top = '50%';
    spinner.style.left = '50%';
    spinner.style.transform = 'translate(-50%, -50%)';
    spinner.style.zIndex = '9999';

    const overlay = document.createElement('div');
    overlay.id = 'spinner-overlay';
    overlay.style.position = 'fixed';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.right = '0';
    overlay.style.bottom = '0';
    overlay.style.background = 'rgba(0,0,0,0.3)';
    overlay.style.zIndex = '9998';

    document.body.appendChild(overlay);
    document.body.appendChild(spinner);
}

function hideSpinner() {
    const spinner = document.getElementById('global-spinner');
    const overlay = document.getElementById('spinner-overlay');
    if (spinner) spinner.remove();
    if (overlay) overlay.remove();
}

// CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);