# Deploy to Hugging Face Spaces (Docker + GitHub Actions)

## One-time HF Space setup

1. Create an empty **Docker SDK** Space under your account (`Settings` tab can pick **Docker** if README does not declare SDK).
2. **Repository secrets (`ANTHROPIC_API_KEY`, `TAVILY_API_KEY`, Langfuse)`** belong in HF: **Space → Settings → Repository secrets**.  
   Secrets stored only on **GitHub** are **not** visible to the running container on HF unless you bake them elsewhere (avoid).
3. Generate a HF **token** with write access (`write`/`repo` scopes as required by HF) for CI to git-push to the Space.

## GitHub repository secrets for CD

Add **`HF_TOKEN`**: HF access token used only by workflow job `deploy_hf` to push `main` to the Space repo.

**(Optional)** **`HF_USERNAME`**: HF account username for git HTTPS (defaults to **`orion211203`** inside the workflow if unset). Set if deploy runs under another user.

## README / Space card YAML (optional)

For the Space card emoji/title/color, prepend a YAML front matter block at the **very top** of `README.md` on the HF branch ([HF docs — Space metadata](https://huggingface.co/docs/hub/spaces-config-reference)).

That duplicates purpose with the GitHub README; alternatively set metadata in HF UI instead of YAML.

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
- Avoid `--force` on `git push` to HF unless you intentionally overwrite history on the Space.
