"""Initial schema for SIM-ONE Framework MCP Server

Revision ID: 001
Revises: 
Create Date: 2024-08-30 20:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable required extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "pg_trgm"')
    
    # Create entities table
    op.create_table(
        'entities',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('uuid', postgresql.UUID(), server_default=sa.text('uuid_generate_v4()'), nullable=False),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('type', sa.Text(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name'),
        sa.UniqueConstraint('uuid')
    )
    
    # Create memories table
    op.create_table(
        'memories',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('uuid', postgresql.UUID(), server_default=sa.text('uuid_generate_v4()'), nullable=False),
        sa.Column('entity_id', sa.Integer(), nullable=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('emotional_state', sa.Text(), nullable=True),
        sa.Column('source_protocol', sa.Text(), nullable=True),
        sa.Column('session_id', sa.Text(), nullable=True),
        sa.Column('emotional_salience', sa.REAL(), server_default='0.5', nullable=False),
        sa.Column('rehearsal_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('last_accessed', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('confidence_score', sa.REAL(), server_default='1.0', nullable=False),
        sa.Column('memory_type', sa.Text(), server_default='episodic', nullable=False),
        sa.Column('actors', postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column('context_tags', postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), server_default='{}', nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.ForeignKeyConstraint(['entity_id'], ['entities.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('uuid')
    )
    
    # Create relationships table
    op.create_table(
        'relationships',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('uuid', postgresql.UUID(), server_default=sa.text('uuid_generate_v4()'), nullable=False),
        sa.Column('source_entity_id', sa.Integer(), nullable=True),
        sa.Column('target_entity_id', sa.Integer(), nullable=True),
        sa.Column('type', sa.Text(), nullable=False),
        sa.Column('strength', sa.REAL(), server_default='1.0', nullable=False),
        sa.Column('metadata', postgresql.JSONB(), server_default='{}', nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.ForeignKeyConstraint(['source_entity_id'], ['entities.id'], ),
        sa.ForeignKeyConstraint(['target_entity_id'], ['entities.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('uuid')
    )
    
    # Create memory_contradictions table
    op.create_table(
        'memory_contradictions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('uuid', postgresql.UUID(), server_default=sa.text('uuid_generate_v4()'), nullable=False),
        sa.Column('memory_id_1', sa.Integer(), nullable=True),
        sa.Column('memory_id_2', sa.Integer(), nullable=True),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('confidence', sa.REAL(), server_default='1.0', nullable=False),
        sa.Column('resolved', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('resolution_notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.ForeignKeyConstraint(['memory_id_1'], ['memories.id'], ),
        sa.ForeignKeyConstraint(['memory_id_2'], ['memories.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('uuid')
    )
    
    # Create performance indexes
    # Entities indexes
    op.create_index('idx_entities_name', 'entities', ['name'], postgresql_using='gin', postgresql_ops={'name': 'gin_trgm_ops'})
    op.create_index('idx_entities_type', 'entities', ['type'])
    op.create_index('idx_entities_created_at', 'entities', ['created_at'])
    
    # Memories indexes
    op.create_index('idx_memories_entity_id', 'memories', ['entity_id'])
    op.create_index('idx_memories_session_id', 'memories', ['session_id'])
    op.create_index('idx_memories_content', 'memories', ['content'], postgresql_using='gin', postgresql_ops={'content': 'gin_trgm_ops'})
    op.create_index('idx_memories_emotional_state', 'memories', ['emotional_state'])
    op.create_index('idx_memories_source_protocol', 'memories', ['source_protocol'])
    op.create_index('idx_memories_emotional_salience', 'memories', [sa.text('emotional_salience DESC')])
    op.create_index('idx_memories_last_accessed', 'memories', [sa.text('last_accessed DESC')])
    op.create_index('idx_memories_memory_type', 'memories', ['memory_type'])
    op.create_index('idx_memories_actors', 'memories', ['actors'], postgresql_using='gin')
    op.create_index('idx_memories_context_tags', 'memories', ['context_tags'], postgresql_using='gin')
    op.create_index('idx_memories_metadata', 'memories', ['metadata'], postgresql_using='gin')
    op.create_index('idx_memories_created_at', 'memories', [sa.text('created_at DESC')])
    
    # Relationships indexes
    op.create_index('idx_relationships_source_entity', 'relationships', ['source_entity_id'])
    op.create_index('idx_relationships_target_entity', 'relationships', ['target_entity_id'])
    op.create_index('idx_relationships_type', 'relationships', ['type'])
    op.create_index('idx_relationships_strength', 'relationships', [sa.text('strength DESC')])
    
    # Contradictions indexes
    op.create_index('idx_contradictions_memory_1', 'memory_contradictions', ['memory_id_1'])
    op.create_index('idx_contradictions_memory_2', 'memory_contradictions', ['memory_id_2'])
    op.create_index('idx_contradictions_resolved', 'memory_contradictions', ['resolved'])
    
    # Create updated_at trigger function
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql'
    """)
    
    # Create triggers for each table
    tables = ['entities', 'memories', 'relationships', 'memory_contradictions']
    for table in tables:
        op.execute(f"""
            CREATE TRIGGER update_{table}_updated_at
                BEFORE UPDATE ON {table}
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column()
        """)


def downgrade() -> None:
    # Drop triggers
    tables = ['entities', 'memories', 'relationships', 'memory_contradictions']
    for table in tables:
        op.execute(f"DROP TRIGGER IF EXISTS update_{table}_updated_at ON {table}")
    
    # Drop trigger function
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column()")
    
    # Drop tables in reverse order (to handle foreign key constraints)
    op.drop_table('memory_contradictions')
    op.drop_table('relationships')
    op.drop_table('memories')
    op.drop_table('entities')
    
    # Drop extensions (optional, might be used by other applications)
    # op.execute('DROP EXTENSION IF EXISTS "pg_trgm"')
    # op.execute('DROP EXTENSION IF EXISTS "uuid-ossp"')