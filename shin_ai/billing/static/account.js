(() => {
  const signIn = document.getElementById('sign-in');
  if (!signIn) return;
  const status = document.getElementById('login-status');
  const link = document.getElementById('link-account');
  const token = new URLSearchParams(window.location.hash.slice(1)).get('token');
  window.history.replaceState(null, '', window.location.pathname);
  if (!token) return;
  signIn.hidden = false;
  const linkChoice = document.getElementById('link-choice');
  if (linkChoice) linkChoice.hidden = false;
  status.textContent = 'Open the account from this private sign-in link.';
  async function exchange(path, payload, pending) {
    signIn.disabled = true;
    if (link) link.disabled = true;
    status.textContent = pending;
    try {
      const response = await fetch(path, {
        method: 'POST', credentials: 'same-origin',
        headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Sign-in failed. Request a new link from the bot.');
      window.location.replace('/');
    } catch (error) {
      status.textContent = error.message || 'Connection failed. Try again.';
      signIn.disabled = false;
      if (link) link.disabled = false;
    }
  }
  signIn.addEventListener('click', () => exchange('/auth/exchange', {token}, 'Signing in…'));
  if (link) link.addEventListener('click', () => exchange('/auth/link', {token, csrf: link.dataset.csrf}, 'Linking accounts…'));
})();
