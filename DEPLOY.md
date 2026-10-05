# Put Gridiron Props online (free, Streamlit Community Cloud)

1. Make a free GitHub account at github.com.
2. Create a new **private** repository named `gridiron-props`.
3. On the repo page: Add file > Upload files. Drag in everything inside the unzipped `gridiron` folder
   (the files, not the folder itself). Commit.
4. Go to share.streamlit.io, sign in with GitHub, click Create app > Deploy a public app from GitHub
   (it will still be private because the repo is private).
5. Repository: `your-name/gridiron-props`, Branch: `main`, Main file: `app.py`.
6. Advanced settings > Secrets, paste:
       ODDS_API_KEY = "your key here"
7. Deploy. First load trains the models (1-2 minutes).

Your record is stored on the app's server and resets when the app restarts after a few days idle.
Use "Download my record" to keep a backup and "Restore a backup" to load it back.

## Updating the app later
On GitHub: Add file > Upload files, drag in the changed files (same names overwrite the old ones),
Commit changes. Streamlit redeploys automatically within a minute or two.
If the app says it's over its resource limits: share.streamlit.io > your app's ⋮ menu > Reboot.
