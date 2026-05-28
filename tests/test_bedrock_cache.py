"""
Tests for AWS Bedrock prompt caching functionality.
"""

import os
from unittest.mock import patch, MagicMock

import pytest

from cnoe_agent_utils import resolve_bedrock_client, uses_anthropic_bedrock_client
from cnoe_agent_utils.llm_factory import LLMFactory


class TestBedrockPromptCaching:
    """Test suite for AWS Bedrock prompt caching support."""

    @pytest.mark.parametrize("client_setting", ["auto", "anthropic", "anthropic-bedrock", "chat-anthropic-bedrock"])
    def test_public_helper_detects_anthropic_bedrock_client(self, client_setting):
        """Test public Bedrock helper used by callers that manage transport clients."""
        with patch.dict(os.environ, {"AWS_BEDROCK_CLIENT": client_setting}):
            assert uses_anthropic_bedrock_client("global.anthropic.claude-sonnet-4-5-v1:0")
            assert LLMFactory.uses_anthropic_bedrock_client("global.anthropic.claude-sonnet-4-5-v1:0")

    @pytest.mark.parametrize("client_setting", ["converse", "legacy"])
    def test_public_helper_respects_explicit_non_anthropic_client(self, client_setting):
        """Test explicit Bedrock client overrides are visible through the public helper."""
        with patch.dict(os.environ, {"AWS_BEDROCK_CLIENT": client_setting}):
            assert not uses_anthropic_bedrock_client("anthropic.claude-sonnet-4-5")

    @patch.dict(os.environ, {"AWS_BEDROCK_CLIENT": "auto"})
    def test_public_resolver_keeps_cache_choice_for_non_anthropic_models(self):
        """Test the public resolver exposes the full cnoe-agent-utils client decision."""
        assert resolve_bedrock_client("amazon.nova-pro-v1:0", enable_cache=True) == "converse"
        assert resolve_bedrock_client("amazon.nova-pro-v1:0", enable_cache=False) == "legacy"
        assert LLMFactory.resolve_bedrock_client("amazon.nova-pro-v1:0", enable_cache=True) == "converse"

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "aws-bedrock",
        "AWS_BEDROCK_MODEL_ID": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret",
        "AWS_BEDROCK_ENABLE_PROMPT_CACHE": "true",
        "AWS_BEDROCK_CLIENT": "auto"
    })
    @patch("langchain_aws.ChatAnthropicBedrock")
    def test_anthropic_model_auto_uses_anthropic_bedrock(self, mock_chat_anthropic_bedrock):
        """Test that Anthropic Bedrock models use ChatAnthropicBedrock by default."""
        mock_instance = MagicMock()
        mock_chat_anthropic_bedrock.return_value = mock_instance

        factory = LLMFactory("aws-bedrock")
        llm = factory.get_llm()

        assert mock_chat_anthropic_bedrock.called
        call_kwargs = mock_chat_anthropic_bedrock.call_args.kwargs
        assert call_kwargs.get("model") == "anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert "model_id" not in call_kwargs
        assert llm == mock_instance

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "aws-bedrock",
        "AWS_BEDROCK_MODEL_ID": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret",
        "AWS_BEDROCK_ENABLE_PROMPT_CACHE": "true",
        "AWS_BEDROCK_CLIENT": "converse"
    })
    @patch("langchain_aws.ChatBedrockConverse")
    def test_explicit_converse_client_overrides_anthropic_auto(self, mock_chatbedrock_converse):
        """Test that AWS_BEDROCK_CLIENT can force ChatBedrockConverse."""
        mock_instance = MagicMock()
        mock_chatbedrock_converse.return_value = mock_instance

        factory = LLMFactory("aws-bedrock")
        llm = factory.get_llm()

        assert mock_chatbedrock_converse.called
        call_kwargs = mock_chatbedrock_converse.call_args.kwargs
        assert call_kwargs.get("model_id") == "anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert llm == mock_instance

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "aws-bedrock",
        "AWS_BEDROCK_MODEL_ID": "us.amazon.nova-premier-v1:0",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret",
        "AWS_BEDROCK_ENABLE_PROMPT_CACHE": "false",
        "AWS_BEDROCK_CLIENT": "auto"
    })
    @patch("langchain_aws.ChatBedrock")
    def test_cache_disabled_uses_chatbedrock(self, mock_chatbedrock):
        """Test that ChatBedrock is used for non-Anthropic models when caching is disabled."""
        mock_instance = MagicMock()
        mock_chatbedrock.return_value = mock_instance

        factory = LLMFactory("aws-bedrock")
        llm = factory.get_llm()

        # Verify ChatBedrock was instantiated
        assert mock_chatbedrock.called
        assert llm == mock_instance

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "aws-bedrock",
        "AWS_BEDROCK_MODEL_ID": "anthropic.claude-3-7-sonnet-20250219",
        "AWS_REGION": "us-west-2",
        "AWS_ACCESS_KEY_ID": "test_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret",
        "AWS_BEDROCK_ENABLE_PROMPT_CACHE": "true",
        "AWS_BEDROCK_CLIENT": "auto"
    })
    @patch("langchain_aws.ChatAnthropicBedrock")
    def test_cache_enabled_log_message(self, mock_chat_anthropic_bedrock, caplog):
        """Test that appropriate log message is shown when caching is enabled."""
        import logging
        caplog.set_level(logging.INFO)

        mock_instance = MagicMock()
        mock_chat_anthropic_bedrock.return_value = mock_instance

        factory = LLMFactory("aws-bedrock")
        llm = factory.get_llm()

        # Check that cache enabled message was logged
        log_messages = [record.message for record in caplog.records]
        assert any("Prompt caching enabled" in msg and "anthropic.claude-3-7-sonnet-20250219" in msg
                   for msg in log_messages), f"Expected cache enabled message in logs: {log_messages}"
        assert any("Using ChatAnthropicBedrock" in msg for msg in log_messages), \
            f"Expected ChatAnthropicBedrock message in logs: {log_messages}"
        assert llm == mock_instance

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "aws-bedrock",
        "AWS_BEDROCK_MODEL_ID": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret",
        "AWS_BEDROCK_ENABLE_PROMPT_CACHE": "true",
        "AWS_BEDROCK_CLIENT": "auto"
    })
    @patch("langchain_aws.ChatAnthropicBedrock")
    def test_regional_anthropic_model_uses_anthropic_bedrock(self, mock_chat_anthropic_bedrock):
        """Test that regional Anthropic model IDs use ChatAnthropicBedrock."""
        mock_instance = MagicMock()
        mock_chat_anthropic_bedrock.return_value = mock_instance

        factory = LLMFactory("aws-bedrock")
        llm = factory.get_llm()

        call_kwargs = mock_chat_anthropic_bedrock.call_args.kwargs
        assert call_kwargs.get("model") == "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert llm == mock_instance

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "aws-bedrock",
        "AWS_BEDROCK_MODEL_ID": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret",
        "AWS_BEDROCK_CLIENT": "auto",
        "AWS_BEDROCK_READ_TIMEOUT": "",
        "AWS_BEDROCK_CONNECT_TIMEOUT": ""
    })
    @patch("langchain_aws.ChatAnthropicBedrock")
    def test_anthropic_model_maps_botocore_config_timeout(self, mock_chat_anthropic_bedrock):
        """Test that Botocore config read timeouts map to Anthropic timeouts."""
        from botocore.config import Config as BotocoreConfig

        mock_instance = MagicMock()
        mock_chat_anthropic_bedrock.return_value = mock_instance

        factory = LLMFactory("aws-bedrock")
        llm = factory.get_llm(config=BotocoreConfig(read_timeout=123, connect_timeout=45))

        call_kwargs = mock_chat_anthropic_bedrock.call_args.kwargs
        assert "config" not in call_kwargs
        assert call_kwargs["timeout"] == 123.0
        assert llm == mock_instance

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "aws-bedrock",
        "AWS_BEDROCK_MODEL_ID": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret",
        "AWS_BEDROCK_CLIENT": "auto",
        "AWS_BEDROCK_READ_TIMEOUT": "",
        "AWS_BEDROCK_CONNECT_TIMEOUT": ""
    })
    @patch("langchain_aws.ChatAnthropicBedrock")
    def test_anthropic_model_maps_shared_boto_client_timeout(self, mock_chat_anthropic_bedrock):
        """Test that shared boto clients are not passed into ChatAnthropicBedrock."""
        from botocore.config import Config as BotocoreConfig

        mock_instance = MagicMock()
        mock_chat_anthropic_bedrock.return_value = mock_instance
        runtime_client = MagicMock()
        runtime_client.meta.config = BotocoreConfig(read_timeout=321, connect_timeout=54)
        control_client = MagicMock()

        factory = LLMFactory("aws-bedrock")
        llm = factory.get_llm(client=runtime_client, bedrock_client=control_client)

        call_kwargs = mock_chat_anthropic_bedrock.call_args.kwargs
        assert "client" not in call_kwargs
        assert "bedrock_client" not in call_kwargs
        assert call_kwargs["timeout"] == 321.0
        assert llm == mock_instance

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "aws-bedrock",
        "AWS_BEDROCK_MODEL_ID": "us.amazon.nova-premier-v1:0",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret",
        "AWS_BEDROCK_ENABLE_PROMPT_CACHE": "true",
        "AWS_BEDROCK_CLIENT": "auto"
    })
    @patch("langchain_aws.ChatBedrockConverse")
    def test_amazon_model_with_caching(self, mock_chatbedrock_converse):
        """Test that Amazon Nova models work with caching enabled."""
        mock_instance = MagicMock()
        mock_chatbedrock_converse.return_value = mock_instance

        factory = LLMFactory("aws-bedrock")
        llm = factory.get_llm()

        # Verify ChatBedrockConverse was called with the full model ID (unchanged)
        call_kwargs = mock_chatbedrock_converse.call_args.kwargs
        assert call_kwargs.get("model_id") == "us.amazon.nova-premier-v1:0"
        assert llm == mock_instance

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "aws-bedrock",
        "AWS_BEDROCK_MODEL_ID": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret",
        "AWS_BEDROCK_PROVIDER": "anthropic",
        "AWS_BEDROCK_CLIENT": "legacy"
    })
    @patch("langchain_aws.ChatBedrock")
    def test_explicit_provider_passed_through(self, mock_chatbedrock):
        """Test that explicitly set AWS_BEDROCK_PROVIDER is passed through."""
        mock_instance = MagicMock()
        mock_chatbedrock.return_value = mock_instance

        factory = LLMFactory("aws-bedrock")
        llm = factory.get_llm()

        # Verify ChatBedrock was called with the explicit provider
        call_kwargs = mock_chatbedrock.call_args.kwargs
        assert call_kwargs.get("provider") == "anthropic"
        assert llm == mock_instance

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "aws-bedrock",
        "AWS_BEDROCK_MODEL_ID": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret",
        "AWS_BEDROCK_CLIENT": "bogus"
    })
    def test_invalid_bedrock_client_raises(self):
        """Test that invalid AWS_BEDROCK_CLIENT values fail fast."""
        factory = LLMFactory("aws-bedrock")

        with pytest.raises(ValueError, match="Unsupported AWS_BEDROCK_CLIENT"):
            factory.get_llm()
