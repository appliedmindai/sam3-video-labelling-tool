# Troubleshooting

## "Pipeline failed" — 401 Cannot access gated repo `facebook/sam3`

```
You are trying to access a gated repo. Make sure to have access to it at
https://huggingface.co/facebook/sam3. 401 Client Error. ...
Access to model facebook/sam3 is restricted. You must have access to it
and be authenticated to access it. Please log in.
```

SAM 3.1 weights are a **gated model** on Hugging Face Hub. The backend downloads them on first use (and may fetch additional files later, even when most of the model is already cached locally — a cached model does not guarantee you'll never see this error). A 401 means the Hub rejected your credentials. Work through these in order:

### 1. Request access to the gated model

Visit [facebook/sam3](https://huggingface.co/facebook/sam3) and accept the model terms. Access is granted per Hugging Face account — without it, even a valid token gets a 401.

### 2. Log in with a token

Create a token with **Read** access at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then:

```bash
hf auth login        # or: huggingface-cli login (older CLI versions)
```

Alternatively, set the `HF_TOKEN` environment variable before starting the backend.

### 3. Expired or revoked token

If you logged in before but suddenly get 401s, your stored token may have expired or been revoked. Verify it:

```bash
python3 -c "from huggingface_hub import whoami; print(whoami())"
```

If this prints `Invalid user token`, re-login with a fresh token:

```bash
hf auth login --force
```

No backend restart is needed — the Hub client reads the token file on each request, so just retry the action in the UI.
