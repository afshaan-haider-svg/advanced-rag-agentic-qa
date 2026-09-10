"use client";

import { useCallback, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const SESSION_ID = "frontend-test-session";

type Citation = {
  citation_number: number;
  document_id?: string;
  filename: string;
  page?: number;
  chunk_index?: number;
  formatted_citation?: string;
  source?: string;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
};

type DocumentItem = {
  document_id: string;
  filename: string;
  file_type?: string;
  total_pages?: number;
  num_chunks?: number;
  status?: string;
};

export default function Home() {
  // -----------------------------
  // Chat states
  // -----------------------------
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState("");

  // -----------------------------
  // History states
  // -----------------------------
  const [historyLoading, setHistoryLoading] = useState(true);

  // -----------------------------
  // Document states
  // -----------------------------
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [documentsLoading, setDocumentsLoading] = useState(true);

  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");
  const [uploadError, setUploadError] = useState("");

  const [deleteLoadingId, setDeleteLoadingId] = useState<string | null>(null);

  // -----------------------------
  // Backend status
  // -----------------------------
  const [backendOnline, setBackendOnline] = useState(false);
  const [healthLoading, setHealthLoading] = useState(true);

  // =========================================================
  // LOAD DOCUMENTS
  // =========================================================
  const loadDocuments = useCallback(async () => {
    setDocumentsLoading(true);

    try {
      const response = await fetch(`${API_URL}/documents`);

      if (!response.ok) {
        throw new Error("Failed to load documents.");
      }

      const data = await response.json();

      if (Array.isArray(data)) {
        setDocuments(data);
      } else if (Array.isArray(data.documents)) {
        setDocuments(data.documents);
      } else {
        setDocuments([]);
      }
    } catch (error) {
      console.error("Document loading error:", error);
      setDocuments([]);
    } finally {
      setDocumentsLoading(false);
    }
  }, []);

  // =========================================================
  // LOAD CHAT HISTORY
  // =========================================================
  const loadChatHistory = useCallback(async () => {
    setHistoryLoading(true);

    try {
      const response = await fetch(
        `${API_URL}/chat/history/${SESSION_ID}`
      );

      if (!response.ok) {
        throw new Error("Failed to load chat history.");
      }

      const data = await response.json();

      const restoredMessages: ChatMessage[] = [];

      (
        data.messages as
          | Array<{
              user_message?: string;
              assistant_answer?: string;
              citations?: Citation[];
              timestamp?: string;
            }>
          | undefined
      )?.forEach((item, index) => {
        if (item.user_message) {
          restoredMessages.push({
            id: `history-user-${index}`,
            role: "user",
            content: item.user_message,
          });
        }

        if (item.assistant_answer) {
          restoredMessages.push({
            id: `history-assistant-${index}`,
            role: "assistant",
            content: item.assistant_answer,
            citations: item.citations || [],
          });
        }
      });

      setMessages(restoredMessages);
    } catch (error) {
      console.error("Chat history error:", error);
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  // =========================================================
  // BACKEND HEALTH
  // =========================================================
  const checkBackendHealth = useCallback(async () => {
    setHealthLoading(true);

    try {
      const response = await fetch(`${API_URL}/health/chat`);

      if (!response.ok) {
        throw new Error("Backend health check failed.");
      }

      const data = await response.json();

      setBackendOnline(
        data.status === "healthy" ||
          data.status === "ok" ||
          response.ok
      );
    } catch (error) {
      console.error("Backend health error:", error);
      setBackendOnline(false);
    } finally {
      setHealthLoading(false);
    }
  }, []);

  // =========================================================
  // INITIAL LOAD
  // =========================================================
  useEffect(() => {
    checkBackendHealth();
    loadDocuments();
    loadChatHistory();
  }, [checkBackendHealth, loadDocuments, loadChatHistory]);

  // =========================================================
  // UPLOAD DOCUMENT
  // =========================================================
  const uploadDocument = async () => {
    if (!selectedFile) return;

    setUploadLoading(true);
    setUploadMessage("");
    setUploadError("");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(`${API_URL}/documents/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Failed to upload document."
        );
      }

      setUploadMessage(
        `${data.filename || selectedFile.name} uploaded successfully. ${
          data.num_chunks ?? 0
        } chunks indexed.`
      );

      setSelectedFile(null);

      await loadDocuments();
    } catch (error) {
      setUploadError(
        error instanceof Error
          ? error.message
          : "Something went wrong while uploading the document."
      );
    } finally {
      setUploadLoading(false);
    }
  };

  // =========================================================
  // DELETE DOCUMENT
  // =========================================================
  const deleteDocument = async (
    documentId: string,
    filename: string
  ) => {
    const confirmed = window.confirm(
      `Delete "${filename}"?\n\nThis will remove the document from the RAG knowledge base.`
    );

    if (!confirmed) return;

    setDeleteLoadingId(documentId);

    try {
      const response = await fetch(
        `${API_URL}/documents/${documentId}`,
        {
          method: "DELETE",
        }
      );

      if (!response.ok) {
        let detail = "Failed to delete document.";

        try {
          const data = await response.json();
          detail = data.detail || detail;
        } catch {
          // Keep default message.
        }

        throw new Error(detail);
      }

      setDocuments((currentDocuments) =>
        currentDocuments.filter(
          (document) => document.document_id !== documentId
        )
      );

      setUploadMessage(`${filename} deleted successfully.`);
      setUploadError("");
    } catch (error) {
      setUploadError(
        error instanceof Error
          ? error.message
          : "Something went wrong while deleting the document."
      );
    } finally {
      setDeleteLoadingId(null);
    }
  };

  // =========================================================
  // SEND CHAT MESSAGE
  // =========================================================
  const sendMessage = async () => {
    const currentMessage = message.trim();

    if (!currentMessage || chatLoading) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: currentMessage,
    };

    setMessages((previous) => [...previous, userMessage]);

    setMessage("");
    setChatLoading(true);
    setChatError("");

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: currentMessage,
          session_id: SESSION_ID,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Failed to get response from backend."
        );
      }

      const assistantMessage: ChatMessage = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: data.answer || "No answer returned.",
        citations: data.citations || [],
      };

      setMessages((previous) => [
        ...previous,
        assistantMessage,
      ]);
    } catch (error) {
      setChatError(
        error instanceof Error
          ? error.message
          : "Something went wrong while contacting the backend."
      );
    } finally {
      setChatLoading(false);
    }
  };

  // =========================================================
  // ENTER TO SEND
  // =========================================================
  const handleKeyDown = (
    event: React.KeyboardEvent<HTMLTextAreaElement>
  ) => {
    if (
      event.key === "Enter" &&
      !event.shiftKey &&
      !event.nativeEvent.isComposing
    ) {
      event.preventDefault();
      void sendMessage();
    }
  };

  return (
    <main className="min-h-screen bg-slate-950 text-white">
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col px-4 py-6 sm:px-6 lg:px-8 lg:py-10">
        {/* HEADER */}
        <header className="mb-8 border-b border-slate-800 pb-6">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.25em] text-cyan-400 sm:text-sm">
                Advanced RAG System
              </p>

              <h1 className="mt-3 text-3xl font-bold tracking-tight sm:text-4xl">
                Agentic Document QA
              </h1>

              <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-400 sm:text-base">
                Upload documents, retrieve relevant context and
                generate grounded answers using RAG, LangGraph and
                Gemini.
              </p>
            </div>

            <div
              className={`inline-flex w-fit items-center gap-2 rounded-full border px-3 py-2 text-xs font-medium ${
                backendOnline
                  ? "border-emerald-900 bg-emerald-950/40 text-emerald-300"
                  : "border-red-900 bg-red-950/40 text-red-300"
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  backendOnline
                    ? "bg-emerald-400"
                    : "bg-red-400"
                }`}
              />

              {healthLoading
                ? "Checking backend..."
                : backendOnline
                  ? "Backend Online"
                  : "Backend Offline"}
            </div>
          </div>
        </header>

        <div className="grid flex-1 gap-6 lg:grid-cols-[350px_minmax(0,1fr)]">
          {/* ================================================= */}
          {/* LEFT DOCUMENT PANEL */}
          {/* ================================================= */}
          <aside className="h-fit rounded-2xl border border-slate-800 bg-slate-900 p-5 lg:sticky lg:top-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold">
                  Documents
                </h2>

                <p className="mt-1 text-xs text-slate-500">
                  RAG Knowledge Base
                </p>
              </div>

              <span className="rounded-full bg-slate-800 px-2.5 py-1 text-xs text-slate-300">
                {documents.length}
              </span>
            </div>

            <p className="mt-4 text-sm leading-6 text-slate-400">
              Upload PDF, TXT or DOCX documents for indexing and
              retrieval.
            </p>

            {/* FILE PICKER */}
            <label className="mt-5 flex cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-slate-700 px-4 py-8 text-center transition hover:border-cyan-400 hover:bg-slate-800/70">
              <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-xl bg-cyan-400/10 text-lg text-cyan-400">
                +
              </div>

              <span className="max-w-full break-words text-sm font-medium text-slate-200">
                {selectedFile
                  ? selectedFile.name
                  : "Choose a document"}
              </span>

              <span className="mt-1 text-xs text-slate-500">
                {selectedFile
                  ? "Ready to upload"
                  : "PDF • TXT • DOCX"}
              </span>

              <input
                type="file"
                className="hidden"
                accept=".pdf,.txt,.docx"
                onChange={(event) => {
                  const file =
                    event.target.files?.[0] || null;

                  setSelectedFile(file);
                  setUploadMessage("");
                  setUploadError("");
                }}
              />
            </label>

            {selectedFile && (
              <button
                type="button"
                onClick={uploadDocument}
                disabled={uploadLoading}
                className="mt-3 w-full rounded-xl bg-cyan-400 px-4 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {uploadLoading
                  ? "Uploading & Indexing..."
                  : "Upload Document"}
              </button>
            )}

            {/* UPLOAD SUCCESS */}
            {uploadMessage && (
              <div className="mt-4 rounded-xl border border-emerald-900 bg-emerald-950/30 p-3 text-sm leading-6 text-emerald-300">
                {uploadMessage}
              </div>
            )}

            {/* UPLOAD ERROR */}
            {uploadError && (
              <div className="mt-4 rounded-xl border border-red-900 bg-red-950/30 p-3 text-sm leading-6 text-red-300">
                {uploadError}
              </div>
            )}

            {/* DOCUMENT LIST */}
            <div className="mt-7 border-t border-slate-800 pt-5">
              <div className="mb-3 flex items-center justify-between">
                <h3 className="text-sm font-semibold text-slate-200">
                  Uploaded Documents
                </h3>

                <button
                  type="button"
                  onClick={loadDocuments}
                  disabled={documentsLoading}
                  className="text-xs font-medium text-cyan-400 transition hover:text-cyan-300 disabled:opacity-50"
                >
                  Refresh
                </button>
              </div>

              {documentsLoading ? (
                <div className="rounded-xl bg-slate-950 p-4 text-center text-sm text-slate-500">
                  Loading documents...
                </div>
              ) : documents.length === 0 ? (
                <div className="rounded-xl border border-slate-800 bg-slate-950 p-4 text-center">
                  <p className="text-sm text-slate-400">
                    No documents uploaded
                  </p>

                  <p className="mt-1 text-xs text-slate-600">
                    Upload your first document above.
                  </p>
                </div>
              ) : (
                <div className="max-h-[390px] space-y-3 overflow-y-auto pr-1">
                  {documents.map((document) => (
                    <div
                      key={document.document_id}
                      className="rounded-xl border border-slate-800 bg-slate-950 p-3"
                    >
                      <div className="flex items-start gap-3">
                        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-slate-800 text-xs font-bold uppercase text-cyan-400">
                          {document.file_type
                            ?.replace(".", "")
                            .slice(0, 4) || "DOC"}
                        </div>

                        <div className="min-w-0 flex-1">
                          <p
                            className="truncate text-sm font-medium text-slate-200"
                            title={document.filename}
                          >
                            {document.filename}
                          </p>

                          <p className="mt-1 text-xs text-slate-500">
                            {document.num_chunks ?? 0} chunks
                            {document.total_pages
                              ? ` • ${document.total_pages} page${
                                  document.total_pages === 1
                                    ? ""
                                    : "s"
                                }`
                              : ""}
                          </p>
                        </div>
                      </div>

                      <div className="mt-3 flex items-center justify-between border-t border-slate-800 pt-3">
                        <span className="inline-flex items-center gap-1.5 text-xs text-emerald-400">
                          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                          {document.status || "indexed"}
                        </span>

                        <button
                          type="button"
                          onClick={() =>
                            deleteDocument(
                              document.document_id,
                              document.filename
                            )
                          }
                          disabled={
                            deleteLoadingId ===
                            document.document_id
                          }
                          className="rounded-lg border border-red-900/80 px-2.5 py-1.5 text-xs font-medium text-red-400 transition hover:bg-red-950/40 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                          {deleteLoadingId ===
                          document.document_id
                            ? "Deleting..."
                            : "Delete"}
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* BACKEND STATUS */}
            <div className="mt-6 rounded-xl bg-slate-950 p-4">
              <p className="text-xs uppercase tracking-wider text-slate-500">
                System Status
              </p>

              <div className="mt-3 flex items-center justify-between">
                <span className="text-sm text-slate-400">
                  FastAPI Backend
                </span>

                <span
                  className={
                    backendOnline
                      ? "text-xs font-medium text-emerald-400"
                      : "text-xs font-medium text-red-400"
                  }
                >
                  {healthLoading
                    ? "Checking"
                    : backendOnline
                      ? "Connected"
                      : "Offline"}
                </span>
              </div>
            </div>
          </aside>

          {/* ================================================= */}
          {/* CHAT PANEL */}
          {/* ================================================= */}
          <section className="flex min-h-[720px] min-w-0 flex-col overflow-hidden rounded-2xl border border-slate-800 bg-slate-900">
            {/* CHAT HEADER */}
            <div className="border-b border-slate-800 px-5 py-5 sm:px-6">
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <h2 className="text-lg font-semibold">
                    Chat with your documents
                  </h2>

                  <p className="mt-1 text-sm text-slate-400">
                    Grounded answers with document citations.
                  </p>
                </div>

                <div className="flex items-center gap-2 text-xs text-slate-500">
                  <span className="h-2 w-2 rounded-full bg-cyan-400" />
                  Session: frontend-test-session
                </div>
              </div>
            </div>

            {/* MESSAGES */}
            <div className="flex-1 overflow-y-auto p-4 sm:p-6">
              {historyLoading && messages.length === 0 ? (
                <div className="flex min-h-[400px] items-center justify-center">
                  <p className="text-sm text-slate-500">
                    Loading chat history...
                  </p>
                </div>
              ) : messages.length === 0 ? (
                <div className="flex min-h-[400px] items-center justify-center">
                  <div className="max-w-md text-center">
                    <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-cyan-400/10 font-bold text-cyan-400">
                      AI
                    </div>

                    <h3 className="text-xl font-semibold">
                      Ask a document question
                    </h3>

                    <p className="mt-2 text-sm leading-6 text-slate-400">
                      Upload documents and ask questions about their
                      content. Answers will include retrieved source
                      citations.
                    </p>

                    <div className="mt-5 rounded-xl border border-slate-800 bg-slate-950 p-3 text-left text-xs leading-5 text-slate-500">
                      Example: What are the main challenges of
                      Artificial Intelligence in education?
                    </div>
                  </div>
                </div>
              ) : (
                <div className="space-y-6">
                  {messages.map((chatMessage) => (
                    <div
                      key={chatMessage.id}
                      className={
                        chatMessage.role === "user"
                          ? "ml-auto max-w-[88%] sm:max-w-[80%]"
                          : "mr-auto max-w-[95%]"
                      }
                    >
                      {chatMessage.role === "user" ? (
                        <div className="rounded-2xl rounded-br-md bg-cyan-400 px-4 py-3 text-slate-950 sm:px-5 sm:py-4">
                          <p className="mb-1 text-[11px] font-bold uppercase tracking-wider">
                            You
                          </p>

                          <p className="whitespace-pre-wrap text-sm leading-6 sm:text-base sm:leading-7">
                            {chatMessage.content}
                          </p>
                        </div>
                      ) : (
                        <div>
                          {/* AI ANSWER */}
                          <div className="rounded-2xl rounded-bl-md border border-slate-700 bg-slate-950 p-4 sm:p-5">
                            <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-cyan-400">
                              AI Answer
                            </p>

                            <div className="text-sm leading-7 text-slate-200 sm:text-base">
  <ReactMarkdown
    components={{
  p: ({ children }) => (
    <p className="mb-3 last:mb-0">{children}</p>
  ),
  strong: ({ children }) => (
    <strong className="font-semibold text-white">
      {children}
    </strong>
  ),
  ul: ({ children }) => (
    <ul className="mb-3 ml-5 list-disc space-y-2">
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className="mb-3 ml-5 list-decimal space-y-2">
      {children}
    </ol>
  ),
  li: ({ children }) => (
    <li className="pl-1">{children}</li>
  ),
  code: ({ children }) => (
    <code className="rounded bg-slate-800 px-1.5 py-0.5 text-cyan-300">
      {children}
    </code>
  ),
}}
  >
    {chatMessage.content}
  </ReactMarkdown>
</div>
                          </div>

                          {/* CITATIONS */}
                          {chatMessage.citations &&
                            chatMessage.citations.length > 0 && (
                              <div className="mt-3 rounded-xl border border-slate-800 bg-slate-950 p-4">
                                <div className="mb-3 flex items-center justify-between">
                                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                                    Sources
                                  </p>

                                  <span className="text-xs text-slate-600">
                                    {
                                      chatMessage.citations
                                        .length
                                    }{" "}
                                    retrieved
                                  </span>
                                </div>

                                <div className="space-y-2">
                                  {chatMessage.citations.map(
                                    (citation, index) => (
                                      <div
                                        key={`${chatMessage.id}-citation-${index}`}
                                        className="rounded-lg border border-slate-800 bg-slate-900 px-3 py-2.5"
                                      >
                                        <p className="break-words text-sm text-slate-300">
                                          {citation.formatted_citation ||
                                            `[${
                                              citation.citation_number
                                            }] ${
                                              citation.filename
                                            }${
                                              citation.page
                                                ? ` — Page ${citation.page}`
                                                : ""
                                            }`}
                                        </p>

                                        {citation.chunk_index !==
                                          undefined && (
                                          <p className="mt-1 text-[11px] text-slate-600">
                                            Retrieved chunk{" "}
                                            {
                                              citation.chunk_index
                                            }
                                          </p>
                                        )}
                                      </div>
                                    )
                                  )}
                                </div>
                              </div>
                            )}
                        </div>
                      )}
                    </div>
                  ))}

                  {/* THINKING */}
                  {chatLoading && (
                    <div className="mr-auto max-w-[90%]">
                      <div className="rounded-2xl rounded-bl-md border border-slate-700 bg-slate-950 px-5 py-4">
                        <div className="flex items-center gap-3">
                          <div className="flex gap-1">
                            <span className="h-1.5 w-1.5 rounded-full bg-cyan-400" />
                            <span className="h-1.5 w-1.5 rounded-full bg-cyan-400 opacity-60" />
                            <span className="h-1.5 w-1.5 rounded-full bg-cyan-400 opacity-30" />
                          </div>

                          <p className="text-sm text-slate-400">
                            LangGraph agent is thinking...
                          </p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* CHAT ERROR */}
                  {chatError && (
                    <div className="rounded-xl border border-red-900 bg-red-950/30 p-4 text-sm leading-6 text-red-300">
                      {chatError}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* CHAT INPUT */}
            <div className="border-t border-slate-800 bg-slate-900 p-4 sm:p-6">
              <div className="rounded-2xl border border-slate-700 bg-slate-950 p-3 transition focus-within:border-cyan-500">
                <textarea
                  value={message}
                  onChange={(event) =>
                    setMessage(event.target.value)
                  }
                  onKeyDown={handleKeyDown}
                  placeholder="Ask a question about your documents..."
                  rows={3}
                  disabled={chatLoading}
                  className="w-full resize-none bg-transparent px-2 py-2 text-sm text-white outline-none placeholder:text-slate-600 disabled:opacity-60"
                />

                <div className="mt-2 flex flex-col gap-3 border-t border-slate-800 pt-3 sm:flex-row sm:items-center sm:justify-between">
                  <span className="text-xs text-slate-500">
                    Enter to send • Shift + Enter for new line
                  </span>

                  <button
                    type="button"
                    onClick={sendMessage}
                    disabled={
                      chatLoading || !message.trim()
                    }
                    className="rounded-xl bg-cyan-400 px-5 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {chatLoading
                      ? "Thinking..."
                      : "Send"}
                  </button>
                </div>
              </div>

              <p className="mt-3 text-center text-xs text-slate-600">
                LangGraph Agentic RAG • Gemini • Persistent
                Vector Search
              </p>
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}