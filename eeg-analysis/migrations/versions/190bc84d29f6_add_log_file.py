"""Add log file

Revision ID: 190bc84d29f6
Revises: 213bcea7f0c7
Create Date: 2025-05-17 21:08:06.745448

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '190bc84d29f6'
down_revision = '213bcea7f0c7'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('logs',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('type', sa.Enum('training', 'testing', name='typeenum'), nullable=False),
    sa.Column('model_type', sa.Enum('nb', 'svm', 'rf', name='modeltypeenum'), nullable=False),
    sa.Column('extraction_type', sa.Enum('time', 'freq', 'both', name='extractiontypeenum'), nullable=False),
    sa.Column('file_path', sa.String(length=255), nullable=True),
    sa.Column('accuracy', sa.String(length=7), nullable=True),
    sa.Column('execution_time', sa.Integer(), nullable=True),
    sa.Column('confusion_matrix', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('logs')
    # ### end Alembic commands ###
