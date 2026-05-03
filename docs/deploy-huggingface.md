# Deploy to Hugging Face Spaces (Docker + GitHub Actions)

## One-time HF Space setup

1. Create an empty **Docker SDK** Space under your account, or reuse an existing Space. The **`README.md` at repo root must start with YAML frontmatter** (see repo `README.md` — `sdk: docker`, optional `app_port: 7860`). Without it, HF shows “Missing configuration in README”.
2. **Repository secrets (`ANTHROPIC_API_KEY`, `TAVILY_API_KEY`, Langfuse)`** belong in HF: **Space → Settings → Repository secrets**.  
   Secrets stored only on **GitHub** are **not** visible to the running container on HF unless you bake them elsewhere (avoid).
3. Generate a HF **token** with write access (`write`/`repo` scopes as required by HF) for CI to git-push to the Space.

## GitHub repository secrets for CD

Add **`HF_TOKEN`**: HF access token used only by workflow job `deploy_hf` to push `main` to the Space repo.

**(Optional)** **`HF_USERNAME`**: HF account username for git HTTPS (defaults to **`orion211203`** inside the workflow if unset). Set if deploy runs under another user.

## Space card metadata (`README.md`)

The **first lines** of root `README.md` must be YAML front matter (title, `sdk: docker`, …) so the Hub can configure the Space ([reference](https://huggingface.co/docs/hub/spaces-config-reference)).  
This repo embeds that block at the top of `README.md` (also used on GitHub; some projects keep a separate README for HF-only — our CD mirrors the same file).

## Local Docker smoke

From repo root:

```bash
docker build -t research-assistant-space .
docker run --rm -p 7860:7860 \
  -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  -e TAVILY_API_KEY="$TAVILY_API_KEY" \
  research-assistant-space
```

## Notes

- First Space build pulls heavy deps (sentence-transformers, chromadb); expect long build minutes.
- Image size grows with PyTorch/embeddings stack; HF free-tier build limits apply.
- **Push rejected (`fetch first`)**? The Space repo may have commits not on GitHub. The **`deploy_hf` job uses `git push --force`** so **GitHub `main` replaces** Space `main`. HF-only edits are overwritten — change code on GitHub.
