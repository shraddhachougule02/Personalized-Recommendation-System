// static/js/filters.js
document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('filters_form');
  const resultsList = document.getElementById('filter_results');
  const statusMsg = document.getElementById('filter_status');
  const clearBtn = document.getElementById('filters_clear');

  if (!form || !resultsList) return; // page might not have filters

  const setStatus = (m) => { if (statusMsg) statusMsg.textContent = m || ''; };

  // ---- Load min/max price placeholders & constraints from backend ----
  (async () => {
    try {
      const res = await fetch('/price_range', { cache: 'no-store' });
      if (!res.ok) return;
      const data = await res.json();
      const min = document.getElementById('price_min');
      const max = document.getElementById('price_max');
      if (min) {
        min.placeholder = String(data.min_price);
        min.min = data.min_price;
      }
      if (max) {
        max.placeholder = String(data.max_price);
        max.max = data.max_price;
      }
    } catch (e) {
      // ignore; inputs will just have no dynamic placeholders
    }
  })();

  // ---- Helpers ----
  const getChecked = (name) =>
    Array.from(document.querySelectorAll(`input[name="${name}"]:checked`)).map(i => i.value);

  const getRadio = (name) => {
    const n = document.querySelector(`input[name="${name}"]:checked`);
    return n ? n.value : null;
  };

  const renderItems = (items) => {
    resultsList.innerHTML = '';
    if (!Array.isArray(items) || items.length === 0) {
      resultsList.innerHTML = '<li>No products found.</li>';
      return;
    }
    const frag = document.createDocumentFragment();
    items.forEach(item => {
      const li = document.createElement('li');
      const strong = document.createElement('strong');
      strong.textContent = item.product_name ?? 'Unknown Product';
      const meta = document.createElement('span');
      const brand = item.brand ?? 'Unknown';
      const price = (item.price != null) ? `₹${item.price}` : 'Price N/A';
      meta.textContent = ` — Brand: ${brand}, Price: ${price}`;
      li.appendChild(strong);
      li.appendChild(meta);
      frag.appendChild(li);
    });
    resultsList.appendChild(frag);
  };

  // ---- Submit (Apply) ----
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    setStatus('Loading…');
    resultsList.innerHTML = '';

    // Required by backend:
    const productType = getRadio('product_type') || 'Cleanser';

    // Backend requires at least one skin_concern; if none selected, send all
    let concerns = getChecked('skin_concern');
    if (concerns.length === 0) {
      concerns = ['Acne', 'Wrinkles', 'Dark Spots', 'Redness'];
    }

    // Optional:
    const skinTypes = getChecked('skin_type');
    const brands = getChecked('brand');
    const priceMinEl = document.getElementById('price_min');
    const priceMaxEl = document.getElementById('price_max');

    const params = new URLSearchParams();
    params.append('product_type', productType);
    concerns.forEach(v => params.append('skin_concern', v));
    skinTypes.forEach(v => params.append('skintype_list', v));
    brands.forEach(v => params.append('brand', v));
    if (priceMinEl && priceMinEl.value) params.append('price_min', priceMinEl.value);
    if (priceMaxEl && priceMaxEl.value) params.append('price_max', priceMaxEl.value);

    try {
      const res = await fetch(`/filter_products?${params.toString()}`, { method: 'GET' });
      if (!res.ok) {
        // 404 from backend means "No products found"
        renderItems([]);
        setStatus('');
        return;
      }
      const data = await res.json();
      renderItems(data);
      setStatus(`${data.length} product(s) found`);
    } catch (err) {
      console.error('Filter error:', err);
      resultsList.innerHTML = '<li>Error loading products.</li>';
      setStatus('');
    } finally {
      // Close any open dropdowns after apply
      document.querySelectorAll('.filters-toolbar details[open]')
        .forEach(d => d.removeAttribute('open'));
    }
  });

  // ---- Clear ----
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      // Reset radios: default to Cleanser
      document.querySelectorAll('input[name="product_type"]').forEach(r => {
        r.checked = (r.value === 'Cleanser');
      });
      // Uncheck all checkboxes
      ['skin_type', 'skin_concern', 'brand'].forEach(name => {
        document.querySelectorAll(`input[name="${name}"]`).forEach(cb => cb.checked = false);
      });
      // Clear prices
      const min = document.getElementById('price_min');
      const max = document.getElementById('price_max');
      if (min) min.value = '';
      if (max) max.value = '';
      // UI
      resultsList.innerHTML = '';
      setStatus('Cleared. Click Apply to load products.');
      // Close open dropdowns
      document.querySelectorAll('.filters-toolbar details[open]')
        .forEach(d => d.removeAttribute('open'));
    });
  }
});
