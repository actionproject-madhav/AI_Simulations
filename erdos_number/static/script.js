document.getElementById('search-btn').addEventListener('click', async () => {
    const start = document.getElementById('start').value;
    const end = document.getElementById('end').value;

    if (!start || !end) {
        alert('Please enter both start and end articles.');
        return;
    }

    // Reset UI
    document.getElementById('results-container').classList.add('hidden');
    document.getElementById('error-container').classList.add('hidden');
    document.getElementById('status-container').classList.remove('hidden');
    document.getElementById('path-list').innerHTML = '';

    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ start, end })
        });
        const data = await response.json();

        if (response.status !== 202) {
            showError(data.error || 'Failed to start search');
            return;
        }

        pollStatus(data.job_id);
    } catch (err) {
        showError('Network error occurred.');
    }
});

function showError(msg) {
    document.getElementById('status-container').classList.add('hidden');
    document.getElementById('error-container').classList.remove('hidden');
    document.getElementById('error-text').innerText = msg;
}

async function pollStatus(jobId) {
    const interval = setInterval(async () => {
        try {
            const response = await fetch(`/api/search/${jobId}`);
            const data = await response.json();

            if (data.status === 'complete') {
                clearInterval(interval);
                showResults(data.path);
            } else if (data.status === 'failed') {
                clearInterval(interval);
                showError(data.error || 'Search failed');
            }
        } catch (err) {
            clearInterval(interval);
            showError('Lost connection to server.');
        }
    }, 1500);
}

function showResults(path) {
    document.getElementById('status-container').classList.add('hidden');
    document.getElementById('results-container').classList.remove('hidden');
    const list = document.getElementById('path-list');
    path.forEach(node => {
        const li = document.createElement('li');
        li.innerText = node;
        list.appendChild(li);
    });
}
