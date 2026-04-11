namespace InferenceWeb;

internal static class WebUiChatPolicy
{
    internal const string ModelSelectionLockedMessage =
        "Use /api/models/load to choose a model before chatting. Changing models during chat is not supported.";

    public static bool TryValidateChatRequest(string requestedModel, string requestedBackend, out string error)
    {
        if (string.IsNullOrWhiteSpace(requestedModel) && string.IsNullOrWhiteSpace(requestedBackend))
        {
            error = null;
            return true;
        }

        error = ModelSelectionLockedMessage;
        return false;
    }
}
