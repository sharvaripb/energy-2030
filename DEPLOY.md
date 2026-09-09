# Deploy on Streamlit Community Cloud

## 1. Create the GitHub repository

1. Sign in to GitHub.
2. Create a new public repository named `energy-2030`.
3. Do not initialize it with a README because this folder already contains one.
4. Upload every file and folder from this repository package.

If you use Git from your computer instead of the GitHub upload screen:

```bash
git init
git add .
git commit -m "Initial ENERGY 2030 portfolio app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/energy-2030.git
git push -u origin main
```

After the repository exists, update `GITHUB_URL` in `src/config.py`, commit and push once more.

## 2. Deploy with Streamlit Community Cloud

1. Go to `https://share.streamlit.io` and sign in with GitHub.
2. Click **Create app**.
3. Choose **Yup, I have an app**.
4. Select your `energy-2030` repository.
5. Branch: `main`.
6. Main file path: `app.py`.
7. Choose a custom app URL if one is available, for example `energy-2030`.
8. Open **Advanced settings** and select Python 3.12.
9. No secrets are required for this project.
10. Click **Deploy**.

Streamlit reads `requirements.txt` automatically and starts the app from the repository root.

## 3. Updating the app

Push changes to the GitHub repository. Streamlit Community Cloud will detect the new commit and redeploy the app automatically.

## 4. If deployment fails

Check the app logs first. The most common causes are:

- A dependency version cannot be installed.
- `app.py` was not selected as the entrypoint.
- The live OWID CSV is temporarily unavailable.
- A file was omitted when the repository was uploaded.

The app intentionally downloads the current OWID Energy CSV at runtime, so it needs outbound internet access.
