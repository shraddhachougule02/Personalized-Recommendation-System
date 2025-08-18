document.addEventListener('DOMContentLoaded', () => {
  const searchBtn = document.getElementById('search_btn');
  const searchInput = document.getElementById('search_query');
  const resultsContainer = document.getElementById('search_results');

  async function performSearch() {
    const query = searchInput.value.trim();
    if (!query) {
      resultsContainer.innerHTML = '<li>Please enter search text.</li>';
      return;
    }

    try {
      const url = new URL('/search', window.location.origin);
      url.searchParams.append('query', query);
      url.searchParams.append('top_k', '10');

      const res = await fetch(url);
      if (!res.ok) throw new Error('Search request failed');
      const data = await res.json();

      resultsContainer.innerHTML = '';
      if (!data.length) {
        resultsContainer.innerHTML = '<li>No results found.</li>';
        return;
      }

      data.forEach(item => {
        const li = document.createElement('li');
        li.textContent = `${item.product_name} - Brand: ${item.brand} - Price: ₹${item.price ?? 'N/A'}`;
        resultsContainer.appendChild(li);
      });
    } catch (err) {
      console.error('Error during search:', err);
      resultsContainer.innerHTML = '<li>Error fetching search results.</li>';
    }
  }

  searchBtn.addEventListener('click', performSearch);
  searchInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      performSearch();
    }
  });
});
