# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability within this project, please send an email to the security team at security@resumematcher.com. All security vulnerabilities will be promptly addressed.

**Please do not report security vulnerabilities through public GitHub issues.**

### What to Include

When reporting a vulnerability, please include:

- Type of vulnerability
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the vulnerability

## Security Best Practices

### 1. Dependency Security

We use `pip-audit` to scan for known vulnerabilities in our dependencies.

```bash
# Run security audit
make security

# Or directly
pip-audit --desc
```

**CI/CD Integration:**
- Security scans run on every push and pull request
- Critical vulnerabilities will fail the build
- Regular dependency updates via Dependabot

### 2. Secret Management

**Never commit secrets to the repository!**

We use multiple layers of secret protection:

#### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Detect secrets before commit
pre-commit run detect-secrets --all-files
```

#### Environment Variables
All sensitive information is stored in environment variables:

```bash
# Example .env file (never commit this!)
MLFLOW_TRACKING_URI=http://localhost:5000
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
DATABASE_URL=postgresql://user:pass@localhost/db
```

#### Secret Scanning
- `detect-secrets` runs on every commit
- GitHub secret scanning enabled
- `.secrets.baseline` tracks known false positives

### 3. Authentication & Authorization

#### API Security
- FastAPI with built-in security features
- HTTPS in production (configure reverse proxy)
- API key authentication (implement if needed)
- Rate limiting (implement if needed)

#### MLflow Security
- Configure authentication for MLflow server
- Use secure storage backend (S3 with IAM)
- Restrict model registry access

### 4. Docker Security

#### Image Security
```dockerfile
# Use specific version tags (not 'latest')
FROM python:3.11-slim

# Run as non-root user
USER appuser

# Minimal attack surface
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
```

#### Container Best Practices
- Scan images with `docker scan` or `trivy`
- Keep base images updated
- Use multi-stage builds to reduce image size
- Don't expose unnecessary ports

### 5. Data Security

#### Data at Rest
- Encrypt sensitive data in storage (S3 encryption)
- Use DVC for data versioning with secure remote storage
- Regular backups with encryption

#### Data in Transit
- Use HTTPS for all API communications
- TLS for database connections
- Secure S3 bucket policies

### 6. Code Security

#### Static Analysis
```bash
# Run linting and security checks
make lint

# Ruff includes security-related checks
ruff check src/ tests/
```

#### Code Review
- All changes require code review
- Security-focused review for authentication/authorization changes
- Automated security checks in CI/CD

### 7. Infrastructure Security

#### Cloud Security (AWS/GCP/Azure)
- Use IAM roles with least privilege
- Enable CloudTrail/Cloud Audit logging
- Configure security groups properly
- Use VPC for network isolation
- Enable encryption at rest

#### Monitoring
- Prometheus metrics for anomaly detection
- Grafana alerts for suspicious activity
- MLflow audit logs
- Application logging (no sensitive data!)

### 8. Compliance

#### GDPR/Privacy
- Personal data handling procedures
- Data retention policies
- Right to deletion implementation
- Privacy by design

#### License Compliance
- All dependencies checked for license compatibility
- MIT license allows broad usage
- Attribution requirements met

## Security Checklist

### Development
- [ ] Pre-commit hooks installed
- [ ] No secrets in code
- [ ] Environment variables for configuration
- [ ] Dependencies up to date
- [ ] Security audit passing

### Before Deployment
- [ ] All tests passing (including security tests)
- [ ] Vulnerability scan completed
- [ ] HTTPS configured
- [ ] Firewall rules configured
- [ ] Monitoring and alerting enabled
- [ ] Backup strategy implemented
- [ ] Incident response plan documented

### Production
- [ ] Regular security updates
- [ ] Monitoring active
- [ ] Logs reviewed regularly
- [ ] Backup tested
- [ ] Access control audited

## Common Vulnerabilities to Avoid

### 1. Injection Attacks
```python
# ❌ BAD: SQL Injection vulnerability
query = f"SELECT * FROM users WHERE name = '{user_input}'"

# ✅ GOOD: Use parameterized queries
query = "SELECT * FROM users WHERE name = ?"
cursor.execute(query, (user_input,))
```

### 2. Sensitive Data Exposure
```python
# ❌ BAD: Logging sensitive data
logger.info(f"User password: {password}")

# ✅ GOOD: Never log sensitive data
logger.info("User authenticated successfully")
```

### 3. Insecure Deserialization
```python
# ❌ BAD: Using pickle with untrusted data
model = pickle.loads(untrusted_data)

# ✅ GOOD: Use safe serialization or validate source
# Use MLflow for model storage with versioning
```

### 4. Using Components with Known Vulnerabilities
```bash
# ✅ GOOD: Regular dependency updates
pip-audit --desc
pip install --upgrade package-name
```

## Security Tools

### Dependency Scanning
- **pip-audit**: Scan Python dependencies for vulnerabilities
- **Safety**: Additional Python security checker
- **Dependabot**: Automated dependency updates

### Secret Detection
- **detect-secrets**: Pre-commit hook for secret detection
- **GitGuardian**: GitHub secret scanning
- **TruffleHog**: Find secrets in git history

### Container Scanning
- **Trivy**: Container vulnerability scanner
- **Snyk**: Security scanning for containers and code
- **Docker Scan**: Built-in Docker security scanning

### Static Analysis
- **Bandit**: Python security linter
- **Ruff**: Fast Python linter with security checks
- **SonarQube**: Comprehensive code quality and security

## Incident Response

### In Case of Security Incident

1. **Identify**: Determine the scope and impact
2. **Contain**: Isolate affected systems
3. **Eradicate**: Remove the vulnerability
4. **Recover**: Restore normal operations
5. **Document**: Record the incident and response
6. **Review**: Post-incident analysis and improvements

### Contact

- **Security Team**: security@resumematcher.com
- **Emergency**: [Emergency contact if critical]

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [Docker Security](https://docs.docker.com/engine/security/)
- [MLflow Security](https://mlflow.org/docs/latest/auth/index.html)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)

---

**Last Updated**: November 2025  
**Next Review**: February 2026

