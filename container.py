from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from core.config import Settings
from core.database import Database
from repositories.memory_repository import MemoryRepository
from repositories.user_repository import UserRepository
from security.passwords import PasswordHasher
from services.ai_service import (
    CrossEncoderReranker,
    FallbackEmbeddingProvider,
    LocalEmbeddingProvider,
    OpenAIEmbeddingProvider,
    OpenAIMetadataGenerator,
    RuleBasedMetadataGenerator,
)
from services.auth_service import AuthService
from services.memory_service import MemoryService
from services.search_service import SearchService
from vectorstore.chroma_store import ChromaVectorStore


class AppContainer:
    def __init__(
        self,
        auth_service: AuthService,
        memory_service: MemoryService,
        search_service: SearchService,
        memory_repository: MemoryRepository,
    ) -> None:
        self.auth_service = auth_service
        self.memory_service = memory_service
        self.search_service = search_service
        self.memory_repository = memory_repository

    @classmethod
    def bootstrap(cls, settings: Settings) -> "AppContainer":
        db = Database(settings.db_path)
        db.migrate()

        user_repository = UserRepository(db)
        memory_repository = MemoryRepository(db)
        password_hasher = PasswordHasher()

        local_embeddings = LocalEmbeddingProvider(settings.local_embedding_model)

        if settings.use_openai:
            client = OpenAI(api_key=settings.openai_api_key)
            embedding_provider = FallbackEmbeddingProvider(
                primary=OpenAIEmbeddingProvider(client, settings.openai_embedding_model),
                fallback=local_embeddings,
            )
            metadata_generator = OpenAIMetadataGenerator(
                client=client,
                model=settings.openai_chat_model,
                fallback=RuleBasedMetadataGenerator(),
            )
        else:
            embedding_provider = local_embeddings
            metadata_generator = RuleBasedMetadataGenerator()

        reranker = CrossEncoderReranker(settings.reranker_model)
        vector_store = ChromaVectorStore(
            path=settings.chroma_path,
            collection_name="recalled_memories",
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        auth_service = AuthService(user_repository, password_hasher)
        memory_service = MemoryService(
            memory_repository=memory_repository,
            metadata_generator=metadata_generator,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            splitter=splitter,
        )
        search_service = SearchService(
            embedding_provider=embedding_provider,
            reranker=reranker,
            vector_store=vector_store,
            vector_top_k=settings.vector_top_k,
            final_top_k=settings.final_top_k,
        )

        return cls(
            auth_service=auth_service,
            memory_service=memory_service,
            search_service=search_service,
            memory_repository=memory_repository,
        )

