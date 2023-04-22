ssh-keygen -t ed25519 -C "jiashu42@sina.com"
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
pbcopy < ~/.ssh/id_ed25519.pub
paste pubkey in github setting



