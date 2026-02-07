# Push this repo to GitHub

## 1. Set your Git identity (one-time, or use `--global`)

```bash
cd /home/harini/armm

git config user.name "Your Name"
git config user.email "your-email@example.com"
```

Use the same email as your GitHub account.

---

## 2. Create the first commit

```bash
git add -A
git status   # check nothing unwanted is staged
git commit -m "Initial commit: English→Hindi speech-to-speech translation (ASR + Marian MT + Piper TTS)"
```

---

## 3. Create a new repo on GitHub

1. Go to **https://github.com/new**
2. Repository name: e.g. **armm** or **en-hi-speech-translation**
3. Choose **Public** (or Private)
4. Do **not** add a README, .gitignore, or license (they already exist locally)
5. Click **Create repository**

---

## 4. Push from your machine

GitHub will show something like:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

Run those in `/home/harini/armm` (replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your repo).

---

## 5. After clone (for you or others)

To get the whisper.cpp submodule:

```bash
git clone --recurse-submodules https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
# or if already cloned:
git submodule update --init --recursive
```

---

## Optional: use SSH instead of HTTPS

If you use SSH keys with GitHub:

```bash
git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```
